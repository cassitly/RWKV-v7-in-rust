//! Argument parsing for the bundled command-line tools.
//!
//! Parsers accept an iterator without the executable name. `Ok(None)` requests
//! help; parsing never prints, exits the process, or opens model/prompt files.

pub mod export;
pub mod inference;

use std::str::FromStr;

fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    match args.next() {
        Some(value) if !value.starts_with("--") => Ok(value),
        _ => Err(format!("{flag} needs a value")),
    }
}

fn number<T: FromStr>(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<T, String> {
    let raw = value(args, flag)?;
    raw.parse()
        .map_err(|_| format!("invalid value for {flag}: '{raw}'"))
}
