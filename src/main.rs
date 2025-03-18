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

#[derive(Debug)]
struct TokenStats {
    total_tokens: usize,
    cost_dollars: f64,
    cost_cached_dollars: f64,
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
            Self::Gpt4o => "Xenova/gpt-4o".to_string(),
            Self::Gemini => "Xenova/gemma-2-tokenizer".to_string(),
            Self::Claude_3_7 => "Xenova/claude-tokenizer".to_string(),
            Self::Claude_3_5 => "Xenova/claude-tokenizer".to_string(),
        }
    }

    fn cost(&self) -> ModelCostInfo {
        match &self {
            Self::Gpt4o => ModelCostInfo {
                input_cost: 2.5,
                output_cost: 10.0,
                cached_input_cost: 1.25,
            },
            Self::Gemini => ModelCostInfo {
                input_cost: 0.10,
                output_cost: 0.40,
                cached_input_cost: 0.025,
            },
            Self::Claude_3_7 => ModelCostInfo {
                input_cost: 3.,
                output_cost: 15.,
                cached_input_cost: 0.3,
            },
            Self::Claude_3_5 => ModelCostInfo {
                input_cost: 3.,
                output_cost: 15.,
                cached_input_cost: 0.3,
            },
        }
    }
}

fn calc_stats(content: &String, tokenizer_type: TokenizerType) -> Result<TokenStats> {
    let tokenizer = Tokenizer::from_pretrained(tokenizer_type.as_hf(), None)
        .map_err(|e| eyre!("Failed to initialize tokenizer: {}", e))?;
    let tokens = tokenizer
        .encode_fast(content.as_str(), false)
        .map_err(|e| eyre!("Failed to encode text: {}", e))?;
    let cost_info = tokenizer_type.cost();
    let tok_count = tokens.len();
    let tok_mils = (tok_count as f64) / 1_000_000.;

    Ok(TokenStats {
        total_tokens: tok_count,
        cost_dollars: cost_info.input_cost * tok_mils,
        cost_cached_dollars: cost_info.cached_input_cost * tok_mils,
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
        "GPT-4o stats: {:#?}",
        calc_stats(&content, TokenizerType::Gpt4o)?
    );
    println!(
        "Gemini (est. via gemma 2) stats: {:#?}",
        calc_stats(&content, TokenizerType::Gemini)?
    );
    println!(
        "Claude 3.7 stats: {:#?}",
        calc_stats(&content, TokenizerType::Claude_3_7)?
    );
    println!(
        "Claude 3.5 stats: {:#?}",
        calc_stats(&content, TokenizerType::Claude_3_5)?
    );
    Ok(())
}
