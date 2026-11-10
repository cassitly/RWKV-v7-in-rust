//! ZIP writer for `torch.save`-compatible archives.
//!
//! Entries are stored uncompressed, and payloads are padded to a 64-byte
//! boundary the same way PyTorch's own writer does — `torch.load(..., mmap=True)`
//! maps records straight out of the file, so misaligned data breaks the consumer
//! even though the archive would still be a valid zip.
//!
//! Data is streamed in with `write_stream`, so a multi-GB tensor never has to be
//! resident.

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};

const LOCAL_SIG: u32 = 0x0403_4b50;
const CDIR_SIG: u32 = 0x0201_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD64_SIG: u32 = 0x0606_4b50;
const EOCD64_LOC_SIG: u32 = 0x0706_4b50;

/// PyTorch aligns record payloads to 64 bytes.
const ALIGN: u64 = 64;
/// Arbitrary extra-field tag used purely as padding; readers skip unknown tags.
const PAD_TAG: u16 = 0x9986;

const ZIP64_THRESHOLD: u64 = 0xFFFF_FFFF;

struct Entry {
    name: String,
    crc: u32,
    size: u64,
    local_header_offset: u64,
}

pub struct ZipWriter {
    w: BufWriter<File>,
    pos: u64,
    entries: Vec<Entry>,
}

/// CRC-32 (IEEE), computed on the fly so payloads need not be buffered.
struct Crc32(u32);

impl Crc32 {
    fn new() -> Self {
        Crc32(0xFFFF_FFFF)
    }
    fn update(&mut self, buf: &[u8]) {
        let mut c = self.0;
        for &b in buf {
            c ^= b as u32;
            for _ in 0..8 {
                // 0xEDB88320 is the reversed IEEE polynomial.
                c = (c >> 1) ^ (0xEDB8_8320 & (!(c & 1)).wrapping_add(1));
            }
        }
        self.0 = c;
    }
    fn finish(self) -> u32 {
        !self.0
    }
}

impl ZipWriter {
    pub fn create(path: &str) -> io::Result<Self> {
        Ok(ZipWriter {
            w: BufWriter::with_capacity(1 << 20, File::create(path)?),
            pos: 0,
            entries: Vec::new(),
        })
    }

    fn put(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.w.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        Ok(())
    }

    /// Local file header, with an extra field sized so the payload lands on a
    /// 64-byte boundary. Returns the header offset.
    fn local_header(&mut self, name: &str, crc: u32, size: u64) -> io::Result<u64> {
        let offset = self.pos;
        let name_b = name.as_bytes();

        // The local header has no offset field, so it only needs zip64 when the
        // payload itself overflows 32 bits. (The central directory decides
        // separately, since it also stores the header offset.)
        let local_zip64 = size >= ZIP64_THRESHOLD;
        let z64_len: u64 = if local_zip64 { 20 } else { 0 };

        // Header is 30 bytes + name + extra; size the padding extra field so the
        // payload starts on a 64-byte boundary.
        let base = offset + 30 + name_b.len() as u64 + z64_len;
        let mut pad = (ALIGN - (base % ALIGN)) % ALIGN;
        // An extra field needs at least its own 4-byte tag+len.
        while pad != 0 && pad < 4 {
            pad += ALIGN;
        }

        let stored: u32 = if local_zip64 {
            0xFFFF_FFFF
        } else {
            size as u32
        };

        self.put(&LOCAL_SIG.to_le_bytes())?;
        self.put(&if local_zip64 { 45u16 } else { 20u16 }.to_le_bytes())?; // version needed
        self.put(&0u16.to_le_bytes())?; // flags
        self.put(&0u16.to_le_bytes())?; // method: stored
        self.put(&0u16.to_le_bytes())?; // time
        self.put(&0u16.to_le_bytes())?; // date
        self.put(&crc.to_le_bytes())?;
        self.put(&stored.to_le_bytes())?; // compressed size
        self.put(&stored.to_le_bytes())?; // uncompressed size
        self.put(&(name_b.len() as u16).to_le_bytes())?;
        self.put(&((z64_len + pad) as u16).to_le_bytes())?;
        self.put(name_b)?;
        if local_zip64 {
            // Zip64 extended information: the two sizes the fixed header sentinelled.
            self.put(&0x0001u16.to_le_bytes())?;
            self.put(&16u16.to_le_bytes())?;
            self.put(&size.to_le_bytes())?; // uncompressed
            self.put(&size.to_le_bytes())?; // compressed
        }
        if pad >= 4 {
            self.put(&PAD_TAG.to_le_bytes())?;
            self.put(&((pad - 4) as u16).to_le_bytes())?;
            self.put(&vec![0u8; (pad - 4) as usize])?;
        }
        debug_assert_eq!(self.pos % ALIGN, 0, "payload must start 64-byte aligned");
        Ok(offset)
    }

    /// Add an entry whose contents are already in memory.
    pub fn write_bytes(&mut self, name: &str, data: &[u8]) -> io::Result<()> {
        let mut crc = Crc32::new();
        crc.update(data);
        let crc = crc.finish();
        let offset = self.local_header(name, crc, data.len() as u64)?;
        self.put(data)?;
        self.entries.push(Entry {
            name: name.to_string(),
            crc,
            size: data.len() as u64,
            local_header_offset: offset,
        });
        Ok(())
    }

    /// Add an entry by streaming exactly `size` bytes out of `src`.
    ///
    /// The CRC is only known after the payload is written, so the header is
    /// patched afterwards by seeking back — cheap, and it keeps memory flat.
    pub fn write_stream<R: Read>(&mut self, name: &str, src: &mut R, size: u64) -> io::Result<()> {
        let offset = self.local_header(name, 0, size)?;
        let crc_field = offset + 14;

        let mut crc = Crc32::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut left = size;
        while left > 0 {
            let n = (left.min(buf.len() as u64)) as usize;
            src.read_exact(&mut buf[..n])?;
            crc.update(&buf[..n]);
            self.put(&buf[..n])?;
            left -= n as u64;
        }
        let crc = crc.finish();

        let end = self.pos;
        self.w.flush()?;
        self.w.seek(SeekFrom::Start(crc_field))?;
        self.w.write_all(&crc.to_le_bytes())?;
        self.w.flush()?;
        self.w.seek(SeekFrom::Start(end))?;
        self.pos = end;

        self.entries.push(Entry {
            name: name.to_string(),
            crc,
            size,
            local_header_offset: offset,
        });
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<()> {
        let cd_start = self.pos;
        let entries: Vec<Entry> = std::mem::take(&mut self.entries);

        for e in &entries {
            let name_b = e.name.as_bytes();
            let big = e.size >= ZIP64_THRESHOLD || e.local_header_offset >= ZIP64_THRESHOLD;
            // Zip64 extended info carries whichever fields overflowed.
            let mut extra: Vec<u8> = Vec::new();
            if big {
                // Field order must match the sentinel order in the fixed header:
                // uncompressed size, compressed size, then local header offset.
                extra.extend_from_slice(&0x0001u16.to_le_bytes());
                extra.extend_from_slice(&24u16.to_le_bytes());
                extra.extend_from_slice(&e.size.to_le_bytes());
                extra.extend_from_slice(&e.size.to_le_bytes());
                extra.extend_from_slice(&e.local_header_offset.to_le_bytes());
            }
            let stored: u32 = if big { 0xFFFF_FFFF } else { e.size as u32 };
            let off32: u32 = if big {
                0xFFFF_FFFF
            } else {
                e.local_header_offset as u32
            };

            self.put(&CDIR_SIG.to_le_bytes())?;
            self.put(&if big { 45u16 } else { 20u16 }.to_le_bytes())?; // version made by
            self.put(&if big { 45u16 } else { 20u16 }.to_le_bytes())?; // version needed
            self.put(&0u16.to_le_bytes())?; // flags
            self.put(&0u16.to_le_bytes())?; // method
            self.put(&0u16.to_le_bytes())?; // time
            self.put(&0u16.to_le_bytes())?; // date
            self.put(&e.crc.to_le_bytes())?;
            self.put(&stored.to_le_bytes())?;
            self.put(&stored.to_le_bytes())?;
            self.put(&(name_b.len() as u16).to_le_bytes())?;
            self.put(&(extra.len() as u16).to_le_bytes())?;
            self.put(&0u16.to_le_bytes())?; // comment len
            self.put(&0u16.to_le_bytes())?; // disk
            self.put(&0u16.to_le_bytes())?; // internal attrs
            self.put(&0u32.to_le_bytes())?; // external attrs
            self.put(&off32.to_le_bytes())?;
            self.put(name_b)?;
            self.put(&extra)?;
        }

        let cd_size = self.pos - cd_start;
        let n = entries.len();
        let need64 = n > 0xFFFE || cd_start >= ZIP64_THRESHOLD || cd_size >= ZIP64_THRESHOLD;

        if need64 {
            let eocd64 = self.pos;
            self.put(&EOCD64_SIG.to_le_bytes())?;
            self.put(&44u64.to_le_bytes())?; // size of remainder
            self.put(&45u16.to_le_bytes())?;
            self.put(&45u16.to_le_bytes())?;
            self.put(&0u32.to_le_bytes())?; // this disk
            self.put(&0u32.to_le_bytes())?; // cd start disk
            self.put(&(n as u64).to_le_bytes())?;
            self.put(&(n as u64).to_le_bytes())?;
            self.put(&cd_size.to_le_bytes())?;
            self.put(&cd_start.to_le_bytes())?;

            self.put(&EOCD64_LOC_SIG.to_le_bytes())?;
            self.put(&0u32.to_le_bytes())?;
            self.put(&eocd64.to_le_bytes())?;
            self.put(&1u32.to_le_bytes())?;
        }

        self.put(&EOCD_SIG.to_le_bytes())?;
        self.put(&0u16.to_le_bytes())?;
        self.put(&0u16.to_le_bytes())?;
        self.put(&(n.min(0xFFFF) as u16).to_le_bytes())?;
        self.put(&(n.min(0xFFFF) as u16).to_le_bytes())?;
        self.put(&(cd_size.min(ZIP64_THRESHOLD) as u32).to_le_bytes())?;
        self.put(&(cd_start.min(ZIP64_THRESHOLD) as u32).to_le_bytes())?;
        self.put(&0u16.to_le_bytes())?; // comment len

        self.w.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_known_vectors() {
        let mut c = Crc32::new();
        c.update(b"123456789");
        assert_eq!(c.finish(), 0xCBF4_3926);
        let mut c = Crc32::new();
        c.update(b"");
        assert_eq!(c.finish(), 0);
        let mut c = Crc32::new();
        c.update(b"The quick brown fox jumps over the lazy dog");
        assert_eq!(c.finish(), 0x414F_A339);
    }
}
