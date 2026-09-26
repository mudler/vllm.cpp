// GLiNER2.5-Decide classification head — the decision head of
// fastino/GLiNER2.5-Decide (MODEL-GLINER25-DECIDE).
//
// Ported from the GLiNER2 library (github.com/fastino-ai/GLiNER2):
//   models/span/model.py   SpanExtractorModel.classifier
//   classification/scoring.py  ClassificationScorer
//
// The classifier is a 3-layer Sequential:
//   Linear(H, 2H) (index 0) → ReLU() (index 1) → Linear(2H, 1) (index 2)
// State-dict keys: classifier.0.weight, classifier.0.bias,
//                  classifier.2.weight, classifier.2.bias
//
// Given label embeddings [num_labels, H] extracted from the encoder output,
// computes per-label logits [num_labels]:
//   hidden = ReLU(label_embs @ w0^T + b0)     // [N, 2H]
//   logits = hidden @ w2^T + b2                  // [N, 1] → [N]
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {
namespace gliner25_decide {

// Classification head weights (F32, PyTorch nn.Linear convention: [out, in]).
struct DecideHeadWeights {
  std::vector<float> w0;  // [2H, H]   classifier.0.weight
  std::vector<float> b0;  // [2H]      classifier.0.bias
  std::vector<float> w2;  // [1, 2H]   classifier.2.weight
  std::vector<float> b2;  // [1]       classifier.2.bias
};

// Classification head params.
struct DecideHeadParams {
  int64_t hidden_size = 1024;  // DeBERTa-v3-large hidden size
  float temperature = 1.0F;     // applied before activation (default 1.0)
};

// Classification head forward: label_embs [N, H] → logits [N].
std::vector<float> ClassifierForward(
    const DecideHeadWeights& hw, const DecideHeadParams& params,
    const std::vector<float>& label_embs, int64_t num_labels);

// Softmax with temperature scaling.
std::vector<float> Softmax(const std::vector<float>& logits, float temperature);

// Sigmoid with temperature scaling.
std::vector<float> Sigmoid(const std::vector<float>& logits, float temperature);

// Encode the question: build the sequence
//   [CLS] text [SEP] [P] task [L] label1 [L] label2 ... [SEP]
// and return token positions of each [L] marker.
struct EncodedSequence {
  std::vector<int32_t> token_ids;        // full sequence
  std::vector<int64_t> label_positions;   // position of each [L] token
};

// Token IDs for special tokens (read from the tokenizer at inference time).
struct SpecialTokens {
  int32_t cls_id = 0;
  int32_t sep_id = 0;
  int32_t p_id = 0;     // [P] prompt token
  int32_t l_id = 0;     // [L] label marker
};

}  // namespace gliner25_decide
}  // namespace vllm
