//! Modified nodal analysis solver

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct NodalAnalysis {
    num_nodes: usize,
    conductance_matrix: Vec<Vec<f64>>,
    current_vector: Vec<f64>,
    voltage_solution: Vec<f64>,
}

impl NodalAnalysis {
    pub fn new(num_nodes: usize) -> Result<Self> {
        Ok(Self {
            num_nodes,
            conductance_matrix: vec![vec![0.0; num_nodes]; num_nodes],
            current_vector: vec![0.0; num_nodes],
            voltage_solution: vec![0.0; num_nodes],
        })
    }

    pub fn solve(&mut self, dt: f64, atp_concentration: f64) -> Result<()> {
        // Simple matrix solution: G*V = I
        // For now, simplified diagonal solution
        for i in 0..self.num_nodes {
            if self.conductance_matrix[i][i] != 0.0 {
                self.voltage_solution[i] = self.current_vector[i] / self.conductance_matrix[i][i];
            }
        }
        Ok(())
    }

    pub fn get_voltages(&self) -> Vec<f64> {
        self.voltage_solution.clone()
    }

    pub fn get_currents(&self) -> Vec<f64> {
        self.current_vector.clone()
    }

    pub fn total_power(&self) -> f64 {
        self.voltage_solution.iter().zip(&self.current_vector)
            .map(|(v, i)| v * i)
            .sum()
    }
}