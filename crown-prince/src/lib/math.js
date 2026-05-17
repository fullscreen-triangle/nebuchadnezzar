import katex from "katex";

const MACROS = {
  "\\Sk": "S_{\\mathrm{k}}",
  "\\St": "S_{\\mathrm{t}}",
  "\\Se": "S_{\\mathrm{e}}",
  "\\Sspace": "\\mathcal{S}",
  "\\Depth": "\\mathcal{M}",
  "\\Drug": "\\mathsf{D}",
  "\\Target": "\\mathsf{T}",
  "\\Patient": "\\mathsf{P}",
  "\\kB": "k_{\\mathrm{B}}",
  "\\Real": "\\mathbb{R}",
  "\\Hol": "\\mathsf{Hol}",
  "\\Iso": "\\mathsf{Iso}",
};

export function renderMath(tex, displayMode = false) {
  try {
    return katex.renderToString(tex, {
      displayMode,
      throwOnError: false,
      strict: "ignore",
      macros: MACROS,
      trust: true,
    });
  } catch (e) {
    return tex;
  }
}
