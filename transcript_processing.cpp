#include "llama.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cctype>
#include "json/single_include/nlohmann/json.hpp"

using json = nlohmann::json;

// Helper function to calculate softmax probabilities from logits
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> softmax_probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (float logit : logits) {
        sum_exp += std::exp(logit - max_logit);
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        softmax_probs[i] = std::exp(logits[i] - max_logit) / sum_exp;
    }
    return softmax_probs;
}

// Helper function to remove non-alphabetic tokens from the logits vector in order to prevent the model 
// from predicting them and "shift the probabilities" (softmax does this) to the remaining tokens
void alphabetic_mask(llama_context* ctx, std::vector<float>* logits_vec) {
    std::vector<bool> mask;
    int n_vocab = llama_n_vocab(ctx);
    for (int token_id = 0; token_id < n_vocab; token_id++) {
        std::string token = llama_token_to_str(ctx, token_id);
        bool mask = true;

        for (char c : token) {
            if (std::isalpha(c) || c == ' ' || c == '\n'|| c == '\'' || c == '\"' || c == '.' || c == ',' || c == '?' || c == '!' || c == ';') {
                mask = false;
                break;
            }
        }
        if (mask) {
            (*logits_vec)[token_id] = std::numeric_limits<float>::lowest();
        }
    }
}

// Helper function to get token count of the sample text. Ignore 
int num_tokens(struct llama_context* ctx, const char* sample_text) {
    int token_count = llama_tokenize(ctx, sample_text, nullptr, 0, false);
    if (token_count <= 0) {
        token_count = -token_count; // Take the absolute value to get the actual number of tokens
    } else {
        std::cerr << "Error: Unexpected positive token count" << std::endl;
        llama_free(ctx);
        return -1;
    }
    return token_count;
}

// Tokenize the prompt using the llama API returns the number of tokens within the prompt
int tokenize_prompt(struct llama_context* ctx, const std::string& prompt, std::vector<llama_token>& context_tokens) {
    
    int token_count = num_tokens(ctx, prompt.c_str());

    // Resize the context_tokens vector to hold the tokens
    context_tokens.resize(token_count);

    // Tokenize the prompt and store the tokens in the context_tokens vector
    int result = llama_tokenize(ctx, prompt.c_str(), context_tokens.data(), token_count, false);
    if (result != token_count) {
        std::cerr << "Error: Tokenization failed" << std::endl;
        llama_free(ctx);
        return -1;
    }
    return token_count;
}

// TODO: Needs to be more throughly tested
// Function to generate contextual embeddings from tokenized text
void generate_embeddings(
    struct llama_context* ctx,
    const std::vector<llama_token>& context_tokens,
    std::vector<float>& embeddings
) {
    // Run inference on the provided context tokens
    int n_tokens = static_cast<int>(context_tokens.size());
    int n_past = 0;
    int n_threads = 3;
    int result = llama_eval(ctx, context_tokens.data(), n_tokens, n_past, n_threads);
    if (result != 0) {
        std::cerr << "Error: llama_eval failed with error code " << result << std::endl;
        exit(-1);
    }
    // Get the embeddings for the input text
    float* raw_embeddings = llama_get_embeddings(ctx);
    if (raw_embeddings == nullptr) {
        std::cerr << "Error: llama_get_embeddings returned nullptr" << std::endl;
        exit(-1);
    }

    // Get the number of dimensions in the embeddings
    int n_embd = llama_n_embd(ctx);

    // Copy the embeddings into the output vector
    embeddings.assign(raw_embeddings, raw_embeddings + n_embd * n_tokens);
}

// Function to generate next-token predictions for a podcast transcription
// resulting in the write of a json file with the predictions and probabilities 
// for the humans, the model, and the ground truth next word. 
void generate_podcast_prediction(
    struct llama_context* ctx,
    const std::string& input_file,
    const std::string& output_file
) {
    std::ifstream infile(input_file);
    std::ofstream outfile(output_file);

    json input_data;
    infile >> input_data;

    json output_data;
    output_data["output_data"] = json::array();

    int entry_num = 0;
    int n_threads = 3;

    // Read each entry within the input data
    for (const auto& entry : input_data["input_data"]) {
        // write to stdout the current line number and post_increment
        std::cout << "Processing entry " << entry_num++ << std::endl;
        
        std::string prompt = entry["prompt"];
        std::string next_word = entry["next_word"];

        // tokenize the prompt and next_word
        std::vector<llama_token> context_tokens;
        tokenize_prompt(ctx, prompt, context_tokens);
        std::vector<llama_token> original_context_tokens = context_tokens;

        std::vector<llama_token> next_word_tokens;
        tokenize_prompt(ctx, next_word, next_word_tokens);

        // std::cout << "prompt and next word tokenized " << std::endl;

        // Run the llama inference to obtain the logits for the next token
        int result = llama_eval(ctx, context_tokens.data(), context_tokens.size(), 0, n_threads);
        if (result != 0) {
            std::cerr << "Inference failed" << std::endl;
        }

        // Complete one prediction step before making a word prediction and assessing the probability of the 
        // true next word

        // Get the logits from the llama API
        float* logits = llama_get_logits(ctx);
        int n_vocab = llama_n_vocab(ctx);

        // Convert logits to a vector and calculate softmax probabilities
        std::vector<float> logits_vec(logits, logits + n_vocab);
        alphabetic_mask(ctx, &logits_vec);
    
        std::vector<float> softmax_probs = softmax(logits_vec);

        // Get the highest probability token and its probability
        int max_index = std::distance(softmax_probs.begin(), std::max_element(softmax_probs.begin(), softmax_probs.end()));
        float max_prob_token = softmax_probs[max_index];
        float max_word_prob = max_prob_token;

        // true next word probability (used after model prediction loop)
        float next_token_prob = softmax_probs[next_word_tokens[0]];

        
        // std::string next_token = llama_token_to_str(ctx, next_word_tokens[0]);
        // std::cout << next_token << " " << next_token_prob << std::endl;
        float next_word_prob = next_token_prob;

        // model word prediction

        // get the str representation of the highest probability token
        std::string max_word = llama_token_to_str(ctx, max_index);

        // std::cout << max_word << " " << max_word_prob << std::endl;

        // std::cout << "entering model prediction loop" << std::endl;

        // get the highest probability word and its probability by iteratively calling the llama_eval function
        // within a while loop that runs until a whitespace char is found within the resulting token str
        while(true) {
            // tokenize max_word and use the resulting tokens as the context tokens for the next llama_eval call
            std::vector<llama_token> max_word_tokens;
            tokenize_prompt(ctx, max_word, max_word_tokens);
            context_tokens.insert(context_tokens.end(), max_word_tokens.begin(), max_word_tokens.end());
            // Run the llama inference to obtain the logits for the next token
            result = llama_eval(ctx, context_tokens.data(), context_tokens.size(), 0, n_threads);
            if (result != 0) {
                std::cerr << "Inference failed" << std::endl;
            }

            // Get the logits from the llama API
            logits = llama_get_logits(ctx);
            n_vocab = llama_n_vocab(ctx);

            // Convert logits to a vector and calculate softmax probabilities
            logits_vec.resize(n_vocab);
            logits_vec.assign(logits, logits + n_vocab);

            softmax_probs.resize(n_vocab);
            softmax_probs = softmax(logits_vec);

            // Get the highest probability token and its probability
            int max_index = std::distance(softmax_probs.begin(), std::max_element(softmax_probs.begin(), softmax_probs.end()));
            float max_prob_token = softmax_probs[max_index];
            std::string token_str = llama_token_to_str(ctx, max_index);
            
            // std::cout << token_str << " " << max_prob_token << std::endl;

            max_word += token_str;
            // break the loop if the new token contains a space, newline, comma, exclamation, question, or semicolon char
            if (token_str.find(' ') != std::string::npos || token_str.find('\n') != std::string::npos ||token_str.find('.') != std::string::npos || token_str.find(',') != std::string::npos || token_str.find('?') != std::string::npos || token_str.find('!') != std::string::npos || token_str.find(';') != std::string::npos){
                break;
            max_word_prob *= max_prob_token;
            }
            
        }

        // std::cout << "exited model prediction loop" << std::endl;

        // obtain the word that is before the breaking char in the max_word str (not the first whitespace char)
        if (max_word[0] == ' ' || max_word[0] == '\n' || max_word[0] == ',' || max_word[0] == '.' || max_word[0] == '?' || max_word[0] == '!' || max_word[0] == ';' || max_word[0] == ':') {
            max_word = max_word.substr(1, max_word.size());
        }
        max_word = max_word.substr(0, max_word.find_first_of(" \n,.:;?!"));

        // restore the original context tokens
        context_tokens = original_context_tokens;

       
        
        // std::cout << "entering true next word probability loop" << std::endl;

        unsigned int i = 1;
        // get the probability of the true next word by iteratively calling the llama_eval function
        while (next_word_tokens.size() > i) {
            // use next word tokens as the context tokens for the next llama_eval call
            std::vector<llama_token> next_word_tokens_temp;
            // collect tokens from start to i
            next_word_tokens_temp = std::vector<llama_token>(next_word_tokens.begin(), next_word_tokens.begin() + i);
            // convert next_tokens_temp to a str using a loop
            std::string next_word_str = "";
            for (size_t j = 0; j < next_word_tokens_temp.size(); j++) {
                next_word_str += llama_token_to_str(ctx, next_word_tokens_temp[j]);
            }
            // std::cout << "next_word_tokens_temp: " << next_word_str << std::endl;
            context_tokens.insert(context_tokens.end(), next_word_tokens_temp.begin(), next_word_tokens_temp.end());
            // Run the llama inference to obtain the logits for the next token
            result = llama_eval(ctx, context_tokens.data(), context_tokens.size(), 0, n_threads);
            if (result != 0) {
                std::cerr << "Inference failed" << std::endl;
            }

            // Get the logits from the llama API
            logits = llama_get_logits(ctx);
            n_vocab = llama_n_vocab(ctx);

            // Convert logits to a vector and calculate softmax probabilities
            logits_vec.resize(n_vocab);
            logits_vec.assign(logits, logits + n_vocab);
            softmax_probs.resize(n_vocab);
            softmax_probs = softmax(logits_vec);

            // use the next word tokens to get the probability of the next token
            next_token_prob = softmax_probs[next_word_tokens[i]];
            // get the str representation of the next token
            std::string next_token = llama_token_to_str(ctx, next_word_tokens[i++]);

            // std::cout << next_token << " " << next_token_prob << std::endl;

            next_word_prob *= next_token_prob;
            
        }
        // std::cout << "exited true next word probability loop" << std::endl;

        // Write the results to a json file

        json output;
        output["prompt"] = prompt;
        output["next_word"] = next_word;
        output["model_prediction"] = max_word;
        output["model_prediction_probability"] = max_word_prob;
        output["next_word_probability"] = next_word_prob;
        // Debug: print output before writing it to the file
        std::cout << "output: " << output.dump(4) << std::endl;
        output_data["output_data"].push_back(output);
    }

    outfile << output_data.dump(4);
}


int main() {
    // Define the path to the Llama model file
    const char* model_path = "models/7B/alpaca-native-7B-ggml/ggml-model-q4_0.bin";
    
    // Define the path to the input and output json files
    // ensure that the input json file has a space in the front of both the input text and the next word
    // to match the llama tokenizer behavior
    const std::string input_json_file = "podcast_testing_in/input1.json";
    const std::string output_json_file = "podcast_testing_out/output1.json";

    // Initialize the the Llama model
    struct llama_context_params params = llama_context_default_params();

    params.embedding = true;

    struct llama_context* ctx = llama_init_from_file(model_path, params);

    // Check if the context is valid
    if (ctx == nullptr) {
        std::cerr << "Error: Failed to initialize Llama context" << std::endl;
        return -1;
    }

    // Call the generate_podcast_prediction function to generate predictions
    generate_podcast_prediction(ctx, input_json_file, output_json_file);

    // Free the llama context
    llama_free(ctx);

    return 0;
}



