//! Just enough ZIP to read a `torch.save` archive.
//!
//! `torch.save` writes entries with method 0 (stored, uncompressed), so no
//! DEFLATE decoder is needed. Entries are read on demand via seek, which keeps
//! peak memory at one tensor rather than the whole checkpoint.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD64_SIG: u32 = 0x0606_4b50;
const EOCD64_LOC_SIG: u32 = 0x0706_4b50;
const CDIR_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;

#[derive(Clone, Copy)]
struct Entry {
    local_header_offset: u64,
    size: u64,
}

pub struct ZipArchive {
    file: File,
    entries: HashMap<String, Entry>,
    names: Vec<String>,
}

fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_u64(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

fn err<T>(msg: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, msg.into()))
}

impl ZipArchive {
    pub fn open(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();

        // The EOCD sits at the very end, possibly followed by a <=64KiB comment.
        let tail_len = file_len.min(66 * 1024) as usize;
        let mut tail = vec![0u8; tail_len];
        file.seek(SeekFrom::End(-(tail_len as i64)))?;
        file.read_exact(&mut tail)?;
        let tail_start = file_len - tail_len as u64;

        let eocd = (0..tail_len.saturating_sub(21))
            .rev()
            .find(|&i| rd_u32(&tail, i) == EOCD_SIG);
        let Some(eocd) = eocd else {
            return err("not a zip archive (no end-of-central-directory record); \
                        is this really a torch.save .pth file?");
        };

        let mut cd_offset = rd_u32(&tail, eocd + 16) as u64;
        let mut n_entries = rd_u16(&tail, eocd + 10) as u64;

        // ZIP64: torch uses it for checkpoints past 4 GiB.
        if cd_offset == 0xFFFF_FFFF || n_entries == 0xFFFF {
            let loc = (0..eocd)
                .rev()
                .find(|&i| rd_u32(&tail, i) == EOCD64_LOC_SIG);
            let Some(loc) = loc else {
                return err("zip64 central directory locator not found");
            };
            let eocd64_off = rd_u64(&tail, loc + 8);
            let mut hdr = [0u8; 56];
            file.seek(SeekFrom::Start(eocd64_off))?;
            file.read_exact(&mut hdr)?;
            if rd_u32(&hdr, 0) != EOCD64_SIG {
                return err("bad zip64 end-of-central-directory signature");
            }
            n_entries = rd_u64(&hdr, 32);
            cd_offset = rd_u64(&hdr, 48);
        }

        // Read the whole central directory in one go; it is small.
        let cd_len = (tail_start + eocd as u64).saturating_sub(cd_offset);
        let mut cd = vec![0u8; cd_len as usize];
        file.seek(SeekFrom::Start(cd_offset))?;
        file.read_exact(&mut cd)?;

        let mut entries = HashMap::new();
        let mut names = Vec::new();
        let mut p = 0usize;
        for _ in 0..n_entries {
            if p + 46 > cd.len() || rd_u32(&cd, p) != CDIR_SIG {
                break;
            }
            let method = rd_u16(&cd, p + 10);
            let mut comp_size = rd_u32(&cd, p + 20) as u64;
            let mut uncomp_size = rd_u32(&cd, p + 24) as u64;
            let name_len = rd_u16(&cd, p + 28) as usize;
            let extra_len = rd_u16(&cd, p + 30) as usize;
            let comment_len = rd_u16(&cd, p + 32) as usize;
            let mut local_off = rd_u32(&cd, p + 42) as u64;

            let name = String::from_utf8_lossy(&cd[p + 46..p + 46 + name_len]).into_owned();
            let extra = &cd[p + 46 + name_len..p + 46 + name_len + extra_len];

            // ZIP64 extended information extra field replaces the 0xFFFFFFFF sentinels.
            let mut e = 0usize;
            while e + 4 <= extra.len() {
                let tag = rd_u16(extra, e);
                let sz = rd_u16(extra, e + 2) as usize;
                let body = &extra[e + 4..(e + 4 + sz).min(extra.len())];
                if tag == 0x0001 {
                    let mut q = 0usize;
                    if uncomp_size == 0xFFFF_FFFF && q + 8 <= body.len() {
                        uncomp_size = rd_u64(body, q);
                        q += 8;
                    }
                    if comp_size == 0xFFFF_FFFF && q + 8 <= body.len() {
                        comp_size = rd_u64(body, q);
                        q += 8;
                    }
                    if local_off == 0xFFFF_FFFF && q + 8 <= body.len() {
                        local_off = rd_u64(body, q);
                    }
                }
                e += 4 + sz;
            }

            if method != 0 && uncomp_size > 0 {
                return err(format!(
                    "zip entry '{name}' uses compression method {method}; only stored (0) is \
                     supported. Re-save with torch.save(..., _use_new_zipfile_serialization=True) \
                     or convert to safetensors."
                ));
            }
            let _ = comp_size;

            entries.insert(
                name.clone(),
                Entry {
                    local_header_offset: local_off,
                    size: uncomp_size,
                },
            );
            names.push(name);
            p += 46 + name_len + extra_len + comment_len;
        }

        Ok(ZipArchive {
            file,
            entries,
            names,
        })
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Byte offset of the entry payload, resolved through its local file header
    /// (the local header's extra field length can differ from the central one).
    /// Returns `(absolute offset, size)`.
    pub fn data_offset(&mut self, name: &str) -> io::Result<(u64, u64)> {
        let Some(e) = self.entries.get(name).copied() else {
            return err(format!("zip entry '{name}' not found"));
        };
        let mut hdr = [0u8; 30];
        self.file.seek(SeekFrom::Start(e.local_header_offset))?;
        self.file.read_exact(&mut hdr)?;
        if rd_u32(&hdr, 0) != LOCAL_SIG {
            return err(format!("bad local file header for '{name}'"));
        }
        let name_len = rd_u16(&hdr, 26) as u64;
        let extra_len = rd_u16(&hdr, 28) as u64;
        Ok((e.local_header_offset + 30 + name_len + extra_len, e.size))
    }

    pub fn read(&mut self, name: &str) -> io::Result<Vec<u8>> {
        let (off, size) = self.data_offset(name)?;
        let mut buf = vec![0u8; size as usize];
        self.file.seek(SeekFrom::Start(off))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read a byte range of an entry without materialising the rest of it.
    pub fn read_range(&mut self, name: &str, start: u64, buf: &mut [u8]) -> io::Result<()> {
        let (off, size) = self.data_offset(name)?;
        if start + buf.len() as u64 > size {
            return err(format!("read past end of zip entry '{name}'"));
        }
        self.file.seek(SeekFrom::Start(off + start))?;
        self.file.read_exact(buf)
    }
}
