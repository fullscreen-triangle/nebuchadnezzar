// 39 reference compounds with NIST CCCBDB vibrational frequencies (cm^-1)
// and rotational constants (cm^-1) for diatomics. Used as both presets and
// as the reference cloud in the S-entropy cube.

export const COMPOUNDS = [
  // Diatomics: { name, formula, type, omega: [...], bRot }
  { name: "H2", formula: "H₂", type: "diatomic", omega: [4401], bRot: 60.85 },
  { name: "D2", formula: "D₂", type: "diatomic", omega: [3115], bRot: 30.44 },
  { name: "N2", formula: "N₂", type: "diatomic", omega: [2330], bRot: 1.99 },
  { name: "O2", formula: "O₂", type: "diatomic", omega: [1580], bRot: 1.45 },
  { name: "F2", formula: "F₂", type: "diatomic", omega: [892], bRot: 0.89 },
  { name: "Cl2", formula: "Cl₂", type: "diatomic", omega: [560], bRot: 0.244 },
  { name: "CO", formula: "CO", type: "diatomic", omega: [2143], bRot: 1.93 },
  { name: "NO", formula: "NO", type: "diatomic", omega: [1875], bRot: 1.7 },
  { name: "HF", formula: "HF", type: "diatomic", omega: [3958], bRot: 20.96 },
  { name: "HCl", formula: "HCl", type: "diatomic", omega: [2886], bRot: 10.59 },
  { name: "HBr", formula: "HBr", type: "diatomic", omega: [2559], bRot: 8.47 },
  { name: "HI", formula: "HI", type: "diatomic", omega: [2230], bRot: 6.51 },

  // Triatomics
  { name: "H2O", formula: "H₂O", type: "triatomic", omega: [1595, 3657, 3756] },
  { name: "CO2", formula: "CO₂", type: "triatomic", omega: [667, 1388, 2349] },
  { name: "SO2", formula: "SO₂", type: "triatomic", omega: [518, 1151, 1362] },
  { name: "NO2", formula: "NO₂", type: "triatomic", omega: [750, 1318, 1618] },
  { name: "O3", formula: "O₃", type: "triatomic", omega: [716, 1043, 1110] },
  { name: "H2S", formula: "H₂S", type: "triatomic", omega: [1183, 2615, 2626] },
  { name: "HCN", formula: "HCN", type: "triatomic", omega: [712, 2097, 3311] },
  { name: "N2O", formula: "N₂O", type: "triatomic", omega: [589, 1285, 2224] },
  { name: "CS2", formula: "CS₂", type: "triatomic", omega: [397, 658, 1535] },
  { name: "OCS", formula: "OCS", type: "triatomic", omega: [520, 859, 2062] },

  // Tetra/pentatomics
  { name: "NH3", formula: "NH₃", type: "tetra", omega: [950, 1627, 3337, 3414] },
  { name: "PH3", formula: "PH₃", type: "tetra", omega: [992, 1118, 2327, 2421] },
  { name: "CH4", formula: "CH₄", type: "tetra", omega: [1306, 1534, 2917, 3019] },
  { name: "CCl4", formula: "CCl₄", type: "tetra", omega: [218, 314, 458, 776] },
  { name: "SiH4", formula: "SiH₄", type: "tetra", omega: [914, 975, 2187, 2191] },
  { name: "CF4", formula: "CF₄", type: "tetra", omega: [435, 632, 909, 1283] },
  {
    name: "H2CO",
    formula: "H₂CO",
    type: "poly",
    omega: [1167, 1249, 1500, 1746, 2782, 2843],
  },

  // Polyatomics
  { name: "C2H2", formula: "C₂H₂", type: "poly", omega: [612, 730, 1974, 3289, 3374] },
  {
    name: "C2H4",
    formula: "C₂H₄",
    type: "poly",
    omega: [810, 943, 949, 1023, 1236, 1342, 1444, 1623, 2989, 3026, 3103, 3106],
  },
  {
    name: "C2H6",
    formula: "C₂H₆",
    type: "poly",
    omega: [289, 822, 995, 1190, 1379, 1468, 1469, 2895, 2954, 2985],
  },
  {
    name: "CH3OH",
    formula: "CH₃OH",
    type: "poly",
    omega: [1033, 1060, 1165, 1345, 1455, 1477, 2844, 2960, 3000, 3681],
  },
  {
    name: "C6H6",
    formula: "C₆H₆",
    type: "poly",
    omega: [606, 992, 1010, 1486, 3068],
  },
  {
    name: "CH3F",
    formula: "CH₃F",
    type: "poly",
    omega: [1049, 1182, 1471, 1467, 2964, 3006],
  },
  {
    name: "CH3Cl",
    formula: "CH₃Cl",
    type: "poly",
    omega: [732, 1017, 1355, 1452, 2937, 3039],
  },
  {
    name: "CH3Br",
    formula: "CH₃Br",
    type: "poly",
    omega: [611, 952, 1306, 1443, 2935, 3056],
  },
  {
    name: "HCOOH",
    formula: "HCOOH",
    type: "poly",
    omega: [638, 1033, 1105, 1229, 1387, 1770, 2943, 3570],
  },
  {
    name: "CH3CN",
    formula: "CH₃CN",
    type: "poly",
    omega: [380, 920, 1041, 1385, 1448, 2267, 2954, 3009],
  },
];

export const TYPE_COLOR = {
  diatomic: "#4477AA",
  triatomic: "#228833",
  tetra: "#CCBB44",
  poly: "#EE6677",
};
