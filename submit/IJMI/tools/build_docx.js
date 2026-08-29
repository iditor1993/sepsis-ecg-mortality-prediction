#!/usr/bin/env node
/* Build a Word manuscript from the IJMI markdown using docx-js.
   Usage:
     node tools/build_docx.js manuscript_IJMI.md documents/IJMI_manuscript.docx [figures|plain]
*/

const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  ExternalHyperlink,
  Footer,
  Header,
  HeadingLevel,
  ImageRun,
  PageBreak,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableLayoutType,
  TableRow,
  TextRun,
  WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const input = path.resolve(ROOT, process.argv[2] || "manuscript_IJMI.md");
const output = path.resolve(ROOT, process.argv[3] || "documents/IJMI_manuscript.docx");
const mode = process.argv[4] || "figures";

const FIGURE_MAP = {
  "Figure 1": "Fig1_cohort_flow.png",
  "Figure 2": "Fig2_analysis_pipeline.png",
  "Figure 3": "Fig3_ROC.png",
  "Figure 4": "Fig4_calibration.png",
  "Figure 5": "Fig5_DCA.png",
  "Figure 6": "Fig6_subgroups.png",
  "Figure 7": "Fig7_SHAP.png",
  "Figure 8": "Fig8_latent_spectrum.png",
  "Graphical abstract": "Graphical_abstract.png",
};

function stripInline(text) {
  return text
    .replace(/\*\*/g, "")
    .replace(/\*/g, "")
    .replace(/__/g, "")
    .replace(/_/g, "")
    .replace(/`/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .trim();
}

function parseInlineRuns(text, base = {}) {
  const runs = [];
  const re = /\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_|`(.+?)`|\[([^\]]+)\]\(([^)]+)\)/g;
  let last = 0;
  let match;
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) {
      runs.push(new TextRun({ text: text.slice(last, match.index), ...base }));
    }
    if (match[1]) {
      runs.push(new TextRun({ text: match[1], bold: true, ...base }));
    } else if (match[2]) {
      runs.push(new TextRun({ text: match[2], italics: true, ...base }));
    } else if (match[3]) {
      runs.push(new TextRun({ text: match[3], bold: true, ...base }));
    } else if (match[4]) {
      runs.push(new TextRun({ text: match[4], italics: true, ...base }));
    } else if (match[5]) {
      runs.push(new TextRun({ text: match[5], font: "Consolas", ...base }));
    } else if (match[6]) {
      runs.push(
        new ExternalHyperlink({
          link: match[7],
          children: [new TextRun({ text: match[6], color: "0563C1", underline: {}, ...base })],
        }),
      );
    }
    last = re.lastIndex;
  }
  if (last < text.length) {
    runs.push(new TextRun({ text: text.slice(last), ...base }));
  }
  return runs.length ? runs : [new TextRun({ text, ...base })];
}

function parseCsvLine(line) {
  const cells = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      cells.push(current.trim());
      current = "";
    } else {
      current += ch;
    }
  }
  cells.push(current.trim());
  return cells;
}

function makeTableFromCells(rows, header = true) {
  const nCols = Math.max(...rows.map((row) => row.length));
  const tableWidth = 9000;
  const width = Math.floor(tableWidth / nCols);
  const widths = Array.from({ length: nCols }, () => width);
  widths[0] += tableWidth - width * nCols;

  const tableRows = rows.map((row, rowIndex) =>
    new TableRow({
      children: Array.from({ length: nCols }, (_, colIndex) => {
        const value = row[colIndex] == null ? "" : String(row[colIndex]).replace(/\|/g, "\\|");
        const isHeader = header && rowIndex === 0;
        return new TableCell({
          width: { size: widths[colIndex], type: WidthType.DXA },
          shading: isHeader
            ? { type: ShadingType.CLEAR, fill: "EEF2F7", color: "auto" }
            : undefined,
          children: [
            new Paragraph({
              alignment: /^\d/.test(value) ? AlignmentType.RIGHT : AlignmentType.LEFT,
              children: parseInlineRuns(value, { bold: isHeader }),
            }),
          ],
        });
      }),
    }),
  );

  return new Table({
    rows: tableRows,
    width: { size: tableWidth, type: WidthType.DXA },
    columnWidths: widths,
    layout: TableLayoutType.FIXED,
  });
}

function readCsvTable(relativePath) {
  const file = path.join(ROOT, relativePath);
  if (!fs.existsSync(file)) {
    return null;
  }
  const rows = fs
    .readFileSync(file, "utf8")
    .trim()
    .split(/\r?\n/)
    .filter((line) => line.trim() !== "")
    .map(parseCsvLine);
  return rows.length ? makeTableFromCells(rows) : null;
}

function addImageParagraph(figureName) {
  const file = path.join(ROOT, "figures", FIGURE_MAP[figureName]);
  if (!fs.existsSync(file)) {
    return [];
  }
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(file),
          transformation: { width: 620, height: 440 },
        }),
      ],
    }),
  ];
}

function parseMarkdown(text, modeName) {
  const lines = text.split(/\r?\n/);
  const children = [];
  let tableBuffer = [];
  let pendingCsv = null;

  function flushTable() {
    if (!tableBuffer.length) return;
    const markdownRows = tableBuffer
      .map((line) => {
        const body = line.startsWith("|") ? line.slice(1) : line;
        const cleaned = body.endsWith("|") ? body.slice(0, -1) : body;
        const protectedLine = cleaned.replace(/\\\|/g, "\u0001");
        return protectedLine.split("|").map((cell) => cell.replace(/\u0001/g, "|").trim());
      })
      .filter((cells) => !cells.every((cell) => /^:?-{2,}:?$/.test(cell.trim())));
    if (markdownRows.length) {
      children.push(makeTableFromCells(markdownRows));
    }
    tableBuffer = [];
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();

    if (trimmed === "") {
      flushTable();
      continue;
    }
    if (trimmed === "---") {
      flushTable();
      children.push(
        new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 1 } },
          children: [],
        }),
      );
      continue;
    }
    if (trimmed.startsWith("|") && trimmed.includes("|")) {
      tableBuffer.push(trimmed);
      continue;
    }
    flushTable();

    if (trimmed.startsWith("# ")) {
      children.push(
        new Paragraph({ heading: HeadingLevel.HEADING_1, children: parseInlineRuns(trimmed.slice(2)) }),
      );
    } else if (trimmed.startsWith("## ")) {
      if (/^## (Tables|Figure legends|References)$/.test(trimmed)) {
        children.push(new Paragraph({ children: [new PageBreak()] }));
      }
      children.push(
        new Paragraph({ heading: HeadingLevel.HEADING_2, children: parseInlineRuns(trimmed.slice(3)) }),
      );
    } else if (trimmed.startsWith("### ")) {
      children.push(
        new Paragraph({ heading: HeadingLevel.HEADING_3, children: parseInlineRuns(trimmed.slice(4)) }),
      );
    } else if (trimmed.startsWith("- ")) {
      children.push(
        new Paragraph({
          bullet: { level: 0 },
          children: parseInlineRuns(trimmed.slice(2)),
        }),
      );
    } else if (/^\*\*(Table|Figure|Graphical abstract)/.test(trimmed)) {
      const caption = stripInline(trimmed);
      children.push(
        new Paragraph({
          spacing: { before: 120, after: 60 },
          children: parseInlineRuns(trimmed),
        }),
      );
      const tableNumber = caption.match(/^Table (\d+)/i);
      if (tableNumber && tableNumber[1] === "1") {
        pendingCsv = "tables/Table1_baseline_characteristics.csv";
      } else if (tableNumber && tableNumber[1] === "6") {
        pendingCsv = "tables/Table6_ECG_availability.csv";
      } else {
        pendingCsv = null;
      }
      if (modeName === "figures") {
        const figureName = caption.match(/^Figure (\d+)/i)
          ? `Figure ${caption.match(/^Figure (\d+)/i)[1]}`
          : caption.startsWith("Graphical abstract")
            ? "Graphical abstract"
            : null;
        if (figureName) {
          children.push(...addImageParagraph(figureName));
        }
      }
    } else {
      children.push(
        new Paragraph({
          spacing: { after: 120 },
          children: parseInlineRuns(trimmed),
        }),
      );
    }

    if (pendingCsv) {
      const csvTable = readCsvTable(pendingCsv);
      if (csvTable) {
        children.push(csvTable);
      }
      pendingCsv = null;
    }
  }
  flushTable();
  return children;
}

async function main() {
  const text = fs.readFileSync(input, "utf8");
  const children = parseMarkdown(text, mode);
  const doc = new Document({
    creator: "Authors",
    title: "IJMI manuscript",
    description: "Submission package",
    styles: {
      default: {
        document: {
          run: { font: "Times New Roman", size: 22 },
          paragraph: { spacing: { line: 360, after: 120 } },
        },
      },
    },
    sections: [
      {
        properties: {
          page: {
            size: { width: 12240, height: 15840 },
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
          },
        },
        headers: {
          default: new Header({
            children: [
              new Paragraph({
                alignment: AlignmentType.RIGHT,
                children: [new TextRun({ text: "IJMI submission", size: 18, color: "666666" })],
              }),
            ],
          }),
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  new TextRun({ text: "Page " }),
                  new TextRun({ children: [PageNumber.CURRENT] }),
                ],
              }),
            ],
          }),
        },
        children,
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(output, buffer);
  console.log(`Wrote ${output}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
