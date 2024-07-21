mod sample_circuit;

use sample_circuit::{LinearRegressionCircuit, create_linear_regression_circuit};

fn main() {
    // Example usage
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let b = 0.5;
    let y = x.iter().zip(w.iter()).map(|(&xi, &wi)| xi * wi).sum::<f64>() + b;

    let circuit = create_linear_regression_circuit(
        x.map(|v| F::from_f64(v)),
        w.map(|v| F::from_f64(v)),
        F::from_f64(b),
        F::from_f64(y),
    );

    // Here you would typically run the circuit proving and verification
    // This depends on the specific ZK proving system you're using
}