use rand::Rng;
use std::f32;

use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct LegacyRPSNetwork {
    pub input_size: usize,
    pub history_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    history: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    hidden: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    probs: Vec<f32>,
}

#[wasm_bindgen]
impl LegacyRPSNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new(
        input_size: usize,
        history_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let history = vec![0.0; input_size * history_size];
        let w1 = (0..input_size * history_size * hidden_size)
            .map(|_| rng.gen::<f32>() * 0.2 - 0.1)
            .collect();
        let b1 = vec![0.0; hidden_size];
        let hidden = vec![0.0; hidden_size];
        let w2 = (0..hidden_size * output_size)
            .map(|_| rng.gen::<f32>() * 0.2 - 0.1)
            .collect();
        let b2 = vec![0.0; output_size];
        let probs = vec![1.0 / (output_size as f32); output_size];

        Self {
            input_size,
            history_size,
            hidden_size,
            output_size,
            w1,
            b1,
            w2,
            b2,
            history,
            hidden,
            probs,
        }
    }

    #[wasm_bindgen]
    pub fn forward(&mut self, input: &[f32]) {
        // Shift history items and add new item
        for i in 1..self.history_size {
            for j in 0..self.input_size {
                self.history[(i - 1) * self.input_size + j] = self.history[i * self.input_size + j]
            }
        }
        for i in 0..self.input_size {
            self.history[(self.history_size - 1) * self.input_size + i] = input[i]
        }

        // Compute hidden layer activations
        for i in 0..self.hidden_size {
            let mut h = 0.0;
            for j in 0..self.input_size * self.history_size {
                h += self.history[j] * self.w1[j * self.hidden_size + i];
            }
            h += self.b1[i];
            self.hidden[i] = h.tanh();
        }

        // Compute output probabilities
        for i in 0..self.output_size {
            let mut o = 0.0;
            for j in 0..self.hidden_size {
                o += self.hidden[j] * self.w2[j * self.output_size + i];
            }
            o += self.b2[i];
            self.probs[i] = o;
        }

        // Apply softmax to output probabilities
        let max_probs = self.probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for i in 0..self.output_size {
            self.probs[i] = (self.probs[i] - max_probs).exp();
            sum += self.probs[i];
        }
        for i in 0..self.output_size {
            self.probs[i] /= sum;
        }
    }

    #[wasm_bindgen]
    pub fn backward(&mut self, label: usize, learning_rate: f32) {
        // Compute the error between the predicted and actual output
        let mut dprobs = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            if i == label {
                dprobs[i] = self.probs[i] - 1.0;
            } else {
                dprobs[i] = self.probs[i];
            }
        }

        // Compute the hidden layer gradient
        let mut dhidden = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let mut dh = 0.0;
            for j in 0..self.output_size {
                dh += self.w2[i * self.output_size + j] * dprobs[j];
            }
            dh *= 1.0 - self.hidden[i] * self.hidden[i];
            dhidden[i] = dh;
        }

        // Update the weights and biases
        for i in 0..self.hidden_size {
            for j in 0..self.output_size {
                self.w2[i * self.output_size + j] -= learning_rate * self.hidden[i] * dprobs[j];
            }
        }
        for i in 0..self.input_size * self.history_size {
            for j in 0..self.hidden_size {
                self.w1[i * self.hidden_size + j] -= learning_rate * self.history[i] * dhidden[j];
            }
        }
        for i in 0..self.hidden_size {
            self.b1[i] -= learning_rate * dhidden[i];
        }
        for i in 0..self.output_size {
            self.b2[i] -= learning_rate * dprobs[i];
        }
    }

    #[wasm_bindgen]
    pub fn probs(&mut self) -> Vec<f32> {
        self.probs.clone()
    }
}
