use clap::Parser;
use color_eyre::{Context as _, Result, eyre::eyre};
use std::{
    io::{self, Read},
    path::PathBuf,
};
use tiktoken_rs::o200k_base;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
struct Args {
    path: Option<PathBuf>,
}

struct TokenStats {
    total_tokens: usize,
    cost_dollars: Option<f64>,
}

struct ModelCostInfo {
    input_cost: f64,
    output_cost: f64,
    cached_input_cost: f64,
}

enum TokenizerType {
    Gpt4o,
    Gemini,
    Claude_3_7,
    Claude_3_5,
}

impl TokenizerType {
    fn as_hf(&self) -> String {
        match &self {
            TokenizerType::Gpt4o => "Xenova/gpt-4o".to_string(),
            TokenizerType::Gemini => "Xenova/gemma-2-tokenizer".to_string(),
            TokenizerType::Claude_3_7 => "Xenova/claude-tokenizer".to_string(),
            TokenizerType::Claude_3_5 => "Xenova/claude-tokenizer".to_string(),
        }
    }

    fn cost(&self) -> ModelCostInfo {
        todo!()
    }
}

fn calc_stats(content: &String, tokenizer_type: TokenizerType) -> Result<TokenStats> {
    let tokenizer = Tokenizer::from_pretrained(tokenizer_type.as_hf(), None)
        .map_err(|e| eyre!("Failed to initialize tokenizer: {}", e))?;
    let tokens = tokenizer
        .encode_fast(content.as_str(), false)
        .map_err(|e| eyre!("Failed to encode text: {}", e))?;
    Ok(TokenStats {
        total_tokens: tokens.len(),
        cost_dollars: None,
    })
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

    println!(
        "GPT-4o token Count: {:?}",
        calc_stats(&content, TokenizerType::Gpt4o)?.total_tokens
    );
    println!(
        "Gemini (est. via gemma 2) token Count: {:?}",
        calc_stats(&content, TokenizerType::Gemini)?.total_tokens
    );
    println!(
        "Claude 3.7 token Count: {:?}",
        calc_stats(&content, TokenizerType::Claude_3_7)?.total_tokens
    );
    println!(
        "Claude 3.5 token Count: {:?}",
        calc_stats(&content, TokenizerType::Claude_3_5)?.total_tokens
    );
    Ok(())
}
