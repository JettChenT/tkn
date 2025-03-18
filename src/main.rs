use clap::Parser;
use color_eyre::{Context as _, Result, eyre::eyre};
use std::{
    io::{self, Read},
    path::PathBuf,
};
use tiktoken_rs::o200k_base;

#[derive(Parser, Debug)]
struct Args {
    path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let content = match args.path {
        Some(path) => std::fs::read_to_string(path)?,
        None => {
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            buffer
        }
    };

    let tokenizer = o200k_base().map_err(|e| eyre!("Failed to initialize tokenizer: {}", e))?;
    let tokens = tokenizer.encode_with_special_tokens(&content);
    println!("Token Count: {:?}", tokens.len());
    Ok(())
}
