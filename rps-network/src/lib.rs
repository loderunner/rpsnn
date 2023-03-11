use ndarray::{arr1, s, Array2};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use std::f32;

use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct RPSNetwork {
    pub input_size: usize,
    pub history_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    w1: Vec<Array2<f32>>,
    b1: Vec<f32>,
    history: Array2<f32>,
    w2: Array2<f32>,
    b2: Array2<f32>,
    hidden: Array2<f32>,
    w3: Array2<f32>,
    b3: Array2<f32>,
    probs: Array2<f32>,
}

#[wasm_bindgen]
impl RPSNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new(
        input_size: usize,
        history_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Self {
        let w1 = (0..history_size)
            .map(|_| Array2::random((input_size, 1), StandardNormal))
            .collect();
        let b1 = vec![0.0; input_size];
        let history = Array2::zeros((history_size, 1));
        let w2 = Array2::random((history_size, hidden_size), StandardNormal);
        let b2 = Array2::zeros((hidden_size, 1));
        let hidden = Array2::zeros((hidden_size, 1));
        let w3 = Array2::random((hidden_size, output_size), StandardNormal);
        let b3 = Array2::zeros((output_size, 1));
        let probs = Array2::from_elem((output_size, 1), 1.0 / (output_size as f32));

        Self {
            input_size,
            history_size,
            hidden_size,
            output_size,
            w1,
            b1,
            history,
            w2,
            b2,
            hidden,
            w3,
            b3,
            probs,
        }
    }

    #[wasm_bindgen]
    pub fn forward(&mut self, input: &[f32]) {
        // Shift history items and add new item
        let past = self.history.slice(s![1.., ..]).to_owned();
        self.history.slice_mut(s![..-1, ..]).assign(&past);
        self.history.slice_mut(s![-1.., ..]).assign(&arr1(input));

        // Compute hidden layer activations
        let a = &self.history.dot(&self.w1);
        let hidden = (a + &self.b1).mapv(|v| v.tanh());
        self.hidden = hidden;

        // Compute output probabilities
        self.probs = &self.hidden * &self.w2 + &self.b2;

        // Apply softmax to output probabilities
        let max_probs = self.probs.max().unwrap().to_owned();
        self.probs.mapv_inplace(|v| (v - max_probs).exp());
        self.probs /= self.probs.sum();
    }

    #[wasm_bindgen]
    pub fn backward(&mut self, label: usize, learning_rate: f32) {
        // Compute the error between the predicted and actual output
        let mut dprobs = self.probs.clone();
        dprobs[(label, 1)] -= 1.0;

        // Compute the hidden layer gradient
        let dhidden = &dprobs * &self.w2 * (1.0 - &self.hidden * &self.hidden);

        // Update the weights and biases
        self.w2 = &self.w2 - &self.hidden * &dprobs;
        self.b2 = &self.b2 - learning_rate * &dprobs;
        self.w1 = &self.w1 - learning_rate * &self.history * &dhidden;
        self.b1 = &self.b1 - learning_rate * &dhidden;
    }

    #[wasm_bindgen]
    pub fn probs(&mut self) -> Vec<f32> {
        self.probs.row(0).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT_SIZE: usize = 3;
    const HISTORY_SIZE: usize = 3;
    const HIDDEN_SIZE: usize = 8;
    const OUTPUT_SIZE: usize = 3;

    #[test]
    fn init_network() {
        let network = RPSNetwork::new(INPUT_SIZE, HISTORY_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        assert_eq!(network.input_size, INPUT_SIZE);
        assert_eq!(network.history_size, HISTORY_SIZE);
        assert_eq!(network.hidden_size, HIDDEN_SIZE);
        assert_eq!(network.output_size, OUTPUT_SIZE);
        assert_eq!(network.w1.shape(), vec![HISTORY_SIZE, HIDDEN_SIZE]);
        assert_eq!(network.b1.shape(), vec![HIDDEN_SIZE, 1]);
        assert_eq!(network.history.shape(), vec![INPUT_SIZE, HISTORY_SIZE]);
        assert_eq!(network.hidden.shape(), vec![HIDDEN_SIZE, 1]);
        assert_eq!(network.w2.shape(), vec![HIDDEN_SIZE, OUTPUT_SIZE]);
        assert_eq!(network.b2.shape(), vec![OUTPUT_SIZE, 1]);
        assert_eq!(network.probs.shape(), vec![OUTPUT_SIZE, 1]);
    }

    #[test]
    fn history() {
        let mut network = RPSNetwork::new(INPUT_SIZE, HISTORY_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        let input = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        network.forward(&input[..3]);
        network.forward(&input[3..6]);
        network.forward(&input[6..]);

        for i in 0..9 {
            assert_eq!(input[i], network.history[(i, 0)]);
        }
    }

    #[test]
    fn forward_pass() {
        let mut network = RPSNetwork::new(INPUT_SIZE, HISTORY_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        let input: Vec<f32> = vec![1.0, 0.0, 0.0];

        network.forward(&input);

        assert!(network.probs.iter().all(|v| v.to_owned() != 0.0));
    }

    #[test]
    fn backward_pass_success() {
        let mut network = RPSNetwork::new(INPUT_SIZE, HISTORY_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        let input: Vec<f32> = vec![1.0, 0.0, 0.0];

        network.forward(&input);

        let paper_prob = network.probs()[1];

        for _ in 0..100 {
            network.backward(1, 0.01);
            network.forward(&input);
        }

        assert!(paper_prob < network.probs()[1]);
    }

    #[test]
    fn backward_pass_fail() {
        let mut network = RPSNetwork::new(INPUT_SIZE, HISTORY_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        let input: Vec<f32> = vec![1.0, 0.0, 0.0];

        network.forward(&input);

        let scissors_prob = network.probs()[2];

        for _ in 0..100 {
            network.backward(1, 0.01);
            network.forward(&input);
        }

        assert!(scissors_prob > network.probs()[2]);
    }
}
