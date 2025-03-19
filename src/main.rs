use clap::Parser;
use color_eyre::{Context as _, Result, eyre::eyre};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{
    fmt::Display,
    io::{self, Read},
    path::PathBuf,
};
use tabled::settings::{Alignment, Style, object::Columns};
use tabled::{Table, Tabled};
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

#[derive(Debug, Tabled)]
struct TokenTableItem<'a> {
    tokenizer: &'a TokenizerType,
    total_tokens: usize,
    cost_dollars: f64,
    cost_cached_dollars: f64,
}

struct ModelCostInfo {
    input_cost: f64,
    output_cost: f64,
    cached_input_cost: f64,
}

#[derive(Debug, Clone)]
enum TokenizerType {
    Gpt4o,
    Gemini,
    Claude_3_7,
    Claude_3_5,
}

impl Display for TokenizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Gpt4o => write!(f, "GPT 4O"),
            Self::Gemini => write!(f, "Gemini"),
            Self::Claude_3_7 => write!(f, "Claude 3.7"),
            Self::Claude_3_5 => write!(f, "Claude 3.5"),
        }
    }
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

fn calc_stats(content: &String, tokenizer_type: &TokenizerType) -> Result<TokenStats> {
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

    let tokenizers = vec![
        TokenizerType::Gpt4o,
        TokenizerType::Gemini,
        TokenizerType::Claude_3_7,
    ];
    let pb = ProgressBar::new(tokenizers.len() as u64);
    let spinner_style = ProgressStyle::with_template("{prefix:.bold.dim} {bar:70.cyan/blue}")
        .unwrap()
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ");
    pb.set_style(spinner_style);
    pb.set_prefix("Tokenizing...");
    let token_stats = tokenizers
        .par_iter()
        .map(|tokenizer_type| calc_stats(&content, tokenizer_type))
        .progress_with(pb)
        .collect::<Result<Vec<TokenStats>>>()?;

    let token_stats_table = token_stats
        .iter()
        .enumerate()
        .map(|(idx, it)| TokenTableItem {
            tokenizer: &tokenizers[idx],
            total_tokens: it.total_tokens,
            cost_dollars: it.cost_dollars,
            cost_cached_dollars: it.cost_cached_dollars,
        });
    let mut table = Table::new(token_stats_table);
    table.with(Style::modern());
    println!("{}", table);
    Ok(())
}
