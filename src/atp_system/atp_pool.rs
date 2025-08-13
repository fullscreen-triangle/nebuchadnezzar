//! ATP Pool Management
//! 
//! Manages adenosine phosphate concentrations and energy charge

use crate::error::{Error, Result};

/// ATP pool managing adenosine phosphate concentrations
#[derive(Debug, Clone)]
pub struct AtpPool {
    atp_concentration: f64,  // mM
    adp_concentration: f64,  // mM
    amp_concentration: f64,  // mM
    total_adenylate: f64,    // mM (conserved)
}

impl AtpPool {
    /// Create new ATP pool with initial concentrations
    pub fn new(atp: f64, adp: f64, amp: f64) -> Result<Self> {
        if atp < 0.0 || adp < 0.0 || amp < 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "nucleotide_concentrations".to_string(),
                value: format!("ATP:{}, ADP:{}, AMP:{}", atp, adp, amp),
            });
        }

        let total_adenylate = atp + adp + amp;
        if total_adenylate <= 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "total_adenylate".to_string(),
                value: total_adenylate.to_string(),
            });
        }

        Ok(Self {
            atp_concentration: atp,
            adp_concentration: adp,
            amp_concentration: amp,
            total_adenylate,
        })
    }

    /// Create physiological ATP pool
    pub fn new_physiological() -> Result<Self> {
        Self::new(5.0, 0.5, 0.05) // Typical cellular concentrations
    }

    /// Get current ATP concentration
    pub fn atp_concentration(&self) -> f64 {
        self.atp_concentration
    }

    /// Get current ADP concentration
    pub fn adp_concentration(&self) -> f64 {
        self.adp_concentration
    }

    /// Get current AMP concentration
    pub fn amp_concentration(&self) -> f64 {
        self.amp_concentration
    }

    /// Get total adenylate pool (should be conserved)
    pub fn total_adenylate(&self) -> f64 {
        self.total_adenylate
    }

    /// Calculate energy charge
    pub fn energy_charge(&self) -> f64 {
        (self.atp_concentration + 0.5 * self.adp_concentration) / 
        (self.atp_concentration + self.adp_concentration + self.amp_concentration)
    }

    /// Consume ATP and produce ADP + Pi
    pub fn consume(&mut self, amount: f64) -> Result<f64> {
        if amount < 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "atp_consumption".to_string(),
                value: amount.to_string(),
            });
        }

        let available = self.atp_concentration.min(amount);
        
        if available <= 0.0 {
            return Err(Error::AtpDepletion);
        }

        self.atp_concentration -= available;
        self.adp_concentration += available;

        Ok(available)
    }

    /// Synthesize ATP from ADP + Pi
    pub fn synthesize(&mut self, amount: f64) -> Result<()> {
        if amount < 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "atp_synthesis".to_string(),
                value: amount.to_string(),
            });
        }

        let available_adp = self.adp_concentration.min(amount);
        
        self.adp_concentration -= available_adp;
        self.atp_concentration += available_adp;

        Ok(())
    }

    /// Add ATP directly (e.g., from external source)
    pub fn add_atp(&mut self, amount: f64) -> Result<()> {
        if amount < 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "atp_addition".to_string(),
                value: amount.to_string(),
            });
        }

        self.atp_concentration += amount;
        self.total_adenylate += amount;

        Ok(())
    }

    /// Convert ADP to AMP (adenylate kinase reaction)
    pub fn adp_to_amp(&mut self, amount: f64) -> Result<()> {
        if amount < 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "adp_conversion".to_string(),
                value: amount.to_string(),
            });
        }

        let available = (self.adp_concentration / 2.0).min(amount);
        
        // 2 ADP -> ATP + AMP
        self.adp_concentration -= 2.0 * available;
        self.atp_concentration += available;
        self.amp_concentration += available;

        Ok(())
    }

    /// Update concentrations maintaining adenylate conservation
    pub fn set_concentrations(&mut self, atp: f64, adp: f64, amp: f64) -> Result<()> {
        if atp < 0.0 || adp < 0.0 || amp < 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "nucleotide_concentrations".to_string(),
                value: format!("ATP:{}, ADP:{}, AMP:{}", atp, adp, amp),
            });
        }

        let new_total = atp + adp + amp;
        let tolerance = 1e-6;
        
        if (new_total - self.total_adenylate).abs() > tolerance {
            return Err(Error::InvalidConfiguration {
                parameter: "adenylate_conservation".to_string(),
                value: format!("Expected: {}, Got: {}", self.total_adenylate, new_total),
            });
        }

        self.atp_concentration = atp;
        self.adp_concentration = adp;
        self.amp_concentration = amp;

        Ok(())
    }

    /// Check if pool is in physiological range
    pub fn is_physiological(&self) -> bool {
        let energy_charge = self.energy_charge();
        energy_charge >= 0.7 && energy_charge <= 0.95 &&
        self.atp_concentration >= 1.0 && self.atp_concentration <= 10.0
    }

    /// Get ATP/ADP ratio (important for metabolic regulation)
    pub fn atp_adp_ratio(&self) -> f64 {
        if self.adp_concentration > 0.0 {
            self.atp_concentration / self.adp_concentration
        } else {
            f64::INFINITY
        }
    }

    /// Calculate free energy of ATP hydrolysis
    pub fn atp_free_energy(&self, temperature: f64, ph: f64) -> f64 {
        // ΔG = ΔG° + RT ln([ADP][Pi]/[ATP])
        // Simplified calculation assuming standard conditions
        let r = 8.314; // J/(mol·K)
        let delta_g_standard = -30500.0; // J/mol at pH 7.0
        
        let ratio = if self.atp_concentration > 0.0 {
            self.adp_concentration / self.atp_concentration
        } else {
            1.0
        };

        delta_g_standard + r * temperature * ratio.ln()
    }
}