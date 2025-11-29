import csv
import html


def csv_to_html(csv_path, html_path, table_title="CSV Table"):
    # Read CSV
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty.")

    headers = rows[0]
    data_rows = rows[1:]

    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(table_title)}</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        background: #fafafa;
        padding: 20px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }}
    th, td {{
        padding: 10px 14px;
        border-bottom: 1px solid #e0e0e0;
        vertical-align: top;
    }}
    th {{
        text-align: left;
        font-weight: 600;
        background: #f5f5f5;
    }}
    tr:hover {{
        background: #f9f9f9;
    }}
    .cell {{
        white-space: pre-wrap; /* preserve newlines in text */
    }}
</style>
</head>
<body>

<h2>{html.escape(table_title)}</h2>

<table>
    <tr>
"""

    # Add header columns
    for h in headers:
        html_content += f"        <th>{html.escape(h)}</th>\n"
    html_content += "    </tr>\n"

    # Add rows
    for row in data_rows:
        html_content += "    <tr>\n"
        for cell in row:
            safe_cell = html.escape(cell)
            html_content += f"        <td class='cell'>{safe_cell}</td>\n"
        html_content += "    </tr>\n"

    # Close HTML
    html_content += """
</table>

</body>
</html>
"""

    # Write file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML file created: {html_path}")


csv_to_html(
    csv_path="emotions.csv",
    html_path="emotions_table.html",
    table_title="Emotion Responses"
)
