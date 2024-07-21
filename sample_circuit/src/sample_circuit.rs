use zkcircuit::{
    circuit::{Circuit, CircuitBuilder},
    field::Field,
    gadgets::{
        fixed_point::FixedPoint,
        lookup::LookupTable,
    },
    witness::Witness,
  };

  const SCALE_FACTOR: u64 = 1 << 16; // 2^16
  const N: usize = 10; // Number of features

  pub struct LinearRegressionCircuit<F: Field> {
      x: [Witness<F>; N],
      w: [Witness<F>; N],
      b: Witness<F>,
      y: Witness<F>,
      scale_lookup: LookupTable<F>,
      unscale_lookup: LookupTable<F>,
  }

  impl<F: Field> Circuit<F> for LinearRegressionCircuit<F> {
      fn synthesize(&self, builder: &mut CircuitBuilder<F>) -> anyhow::Result<()> {
          // 1. Scaling Layer
          let mut scaled_x = [Witness::default(); N];
          let mut scaled_w = [Witness::default(); N];
          for i in 0..N {
              scaled_x[i] = builder.mul(self.x[i], F::from(SCALE_FACTOR));
              scaled_w[i] = builder.mul(self.w[i], F::from(SCALE_FACTOR));
              builder.lookup(&self.scale_lookup, self.x[i], scaled_x[i])?;
              builder.lookup(&self.scale_lookup, self.w[i], scaled_w[i])?;
          }
          let scaled_b = builder.mul(self.b, F::from(SCALE_FACTOR * SCALE_FACTOR));
          builder.lookup(&self.scale_lookup, self.b, scaled_b)?;

          // 2. Inner Product Layer
          let mut z = builder.zero();
          for i in 0..N {
              let product = builder.mul(scaled_x[i], scaled_w[i]);
              z = builder.add(z, product);
          }

          // 3. Bias Addition Layer
          let z_with_bias = builder.add(z, scaled_b);

          // 4. Unscaling Layer
          let y_scaled = builder.div(z_with_bias, F::from(SCALE_FACTOR * SCALE_FACTOR));
          builder.lookup(&self.unscale_lookup, y_scaled, self.y)?;

          // Constraint: Check if y is correctly calculated
          builder.assert_eq(y_scaled, self.y);

          Ok(())
      }
  }

  pub fn create_linear_regression_circuit<F: Field>(
      x: [F; N],
      w: [F; N],
      b: F,
      y: F,
  ) -> LinearRegressionCircuit<F> {
      let mut builder = CircuitBuilder::new();
      let x_witnesses = x.map(|v| builder.witness(v));
      let w_witnesses = w.map(|v| builder.witness(v));
      let b_witness = builder.witness(b);
      let y_witness = builder.witness(y);

      // Create lookup tables for scaling and unscaling
      let scale_lookup = LookupTable::new(|x| x * F::from(SCALE_FACTOR));
      let unscale_lookup = LookupTable::new(|x| x / F::from(SCALE_FACTOR * SCALE_FACTOR));

      LinearRegressionCircuit {
          x: x_witnesses,
          w: w_witnesses,
          b: b_witness,
          y: y_witness,
          scale_lookup,
          unscale_lookup,
      }
  }

