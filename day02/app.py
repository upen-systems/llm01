"""
Day 02 — LLM Cost Calculator Web App
Flask single-page app for estimating LLM API costs across providers.

Run:  python3 app.py
URL:  http://localhost:5002
"""

from flask import Flask, request, render_template_string

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model pricing: (input $/1M tokens, output $/1M tokens)
# ---------------------------------------------------------------------------
MODELS = {
    "claude-sonnet-4":    {"label": "Claude Sonnet 4",    "input": 3.00,  "output": 15.00, "provider": "anthropic"},
    "claude-haiku-4-5":   {"label": "Claude Haiku 4.5",   "input": 0.80,  "output":  4.00, "provider": "anthropic"},
    "gpt-4o":             {"label": "GPT-4o",              "input": 2.50,  "output": 10.00, "provider": "openai"},
    "gpt-4o-mini":        {"label": "GPT-4o mini",         "input": 0.15,  "output":  0.60, "provider": "openai"},
    "gemini-1.5-pro":     {"label": "Gemini 1.5 Pro",      "input": 1.25,  "output":  5.00, "provider": "google"},
    "gemini-1.5-flash":   {"label": "Gemini 1.5 Flash",    "input": 0.075, "output":  0.30, "provider": "google"},
}

# ---------------------------------------------------------------------------
# Cost calculation helpers
# ---------------------------------------------------------------------------

def calc_cost(model_key, monthly_requests, avg_input, avg_output,
              cached_tokens=0, use_caching=False):
    """
    Returns a dict with monthly/annual cost breakdown for one model.

    Prompt caching (Anthropic only): cached input tokens get a 90% discount.
    Non-cached input = avg_input - cached_tokens (clamped to 0).
    """
    m = MODELS[model_key]
    in_price  = m["input"]   / 1_000_000
    out_price = m["output"]  / 1_000_000

    # Effective input tokens per request
    if use_caching and m["provider"] == "anthropic" and cached_tokens > 0:
        uncached = max(avg_input - cached_tokens, 0)
        cached   = min(cached_tokens, avg_input)
        input_cost_per_req = uncached * in_price + cached * in_price * 0.10
    else:
        input_cost_per_req = avg_input * in_price
        cached = 0

    output_cost_per_req = avg_output * out_price
    cost_per_req        = input_cost_per_req + output_cost_per_req

    monthly = cost_per_req * monthly_requests
    annual  = monthly * 12

    return {
        "model_key":    model_key,
        "label":        m["label"],
        "provider":     m["provider"],
        "in_price":     m["input"],
        "out_price":    m["output"],
        "input_cost":   input_cost_per_req * monthly_requests,
        "output_cost":  output_cost_per_req * monthly_requests,
        "cost_per_req": cost_per_req,
        "monthly":      monthly,
        "annual":       annual,
        "cached_tokens": cached if use_caching else 0,
    }


def fmt_usd(value):
    """Format a dollar amount nicely."""
    if value >= 1000:
        return f"${value:,.2f}"
    if value >= 1:
        return f"${value:.4f}"
    return f"${value:.6f}"


# ---------------------------------------------------------------------------
# HTML template (inline — no external dependencies)
# ---------------------------------------------------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Cost Calculator</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 2rem 1rem;
  }

  .container { max-width: 860px; margin: 0 auto; }

  h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.25rem;
  }
  .subtitle { color: #94a3b8; margin-bottom: 2rem; font-size: 0.95rem; }

  /* Card */
  .card {
    background: #1e2130;
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }
  .card h2 {
    font-size: 1rem;
    font-weight: 600;
    color: #93c5fd;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1.25rem;
  }

  /* Form grid */
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .form-group { display: flex; flex-direction: column; gap: 0.4rem; }
  .form-group.full { grid-column: 1 / -1; }
  label { font-size: 0.85rem; color: #94a3b8; font-weight: 500; }
  input[type="text"], input[type="number"], select {
    background: #0f1117;
    border: 1px solid #3d4460;
    border-radius: 8px;
    color: #e2e8f0;
    font-size: 0.95rem;
    padding: 0.55rem 0.75rem;
    outline: none;
    transition: border-color 0.15s;
  }
  input:focus, select:focus { border-color: #6366f1; }
  .hint { font-size: 0.75rem; color: #64748b; }

  /* Checkbox row */
  .checkbox-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.75rem; }
  .checkbox-row input[type="checkbox"] { width: 16px; height: 16px; accent-color: #6366f1; cursor: pointer; }
  .checkbox-row label { font-size: 0.9rem; color: #e2e8f0; cursor: pointer; margin: 0; }
  #caching-extra { display: none; margin-top: 0.75rem; }

  /* Submit */
  button[type="submit"] {
    margin-top: 1rem;
    width: 100%;
    background: #6366f1;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    padding: 0.75rem;
    cursor: pointer;
    transition: background 0.15s;
  }
  button[type="submit"]:hover { background: #4f46e5; }

  /* Results */
  .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem; }
  .stat-box {
    background: #0f1117;
    border: 1px solid #2d3348;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }
  .stat-box .value { font-size: 1.5rem; font-weight: 700; color: #a5f3fc; }
  .stat-box .label { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }

  /* Warning banner */
  .warning {
    background: #422006;
    border: 1px solid #c2410c;
    border-radius: 8px;
    color: #fed7aa;
    font-size: 0.88rem;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
  }

  /* Breakdown bar */
  .breakdown { margin: 1rem 0; }
  .bar-wrap { background: #0f1117; border-radius: 6px; height: 18px; overflow: hidden; display: flex; }
  .bar-input  { background: #6366f1; height: 100%; transition: width 0.4s; }
  .bar-output { background: #f472b6; height: 100%; transition: width 0.4s; }
  .bar-legend { display: flex; gap: 1.5rem; margin-top: 0.5rem; font-size: 0.8rem; color: #94a3b8; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
  .dot-input  { background: #6366f1; }
  .dot-output { background: #f472b6; }

  /* Comparison table */
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th {
    background: #0f1117;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    padding: 0.6rem 0.75rem;
    text-align: right;
  }
  th:first-child { text-align: left; }
  td { padding: 0.55rem 0.75rem; border-top: 1px solid #1e2130; text-align: right; }
  td:first-child { text-align: left; color: #e2e8f0; }
  tr.selected { background: #1e293b; }
  tr.selected td:first-child { color: #a5f3fc; font-weight: 600; }
  .badge {
    display: inline-block;
    font-size: 0.65rem;
    padding: 1px 6px;
    border-radius: 4px;
    margin-left: 6px;
    font-weight: 600;
    vertical-align: middle;
  }
  .badge-anthropic { background: #312e81; color: #a5b4fc; }
  .badge-openai    { background: #064e3b; color: #6ee7b7; }
  .badge-google    { background: #422006; color: #fbbf24; }

  /* Caching savings */
  .savings-box {
    background: #0f2218;
    border: 1px solid #166534;
    border-radius: 10px;
    padding: 1rem 1.25rem;
  }
  .savings-box .title { color: #4ade80; font-weight: 600; margin-bottom: 0.5rem; }
  .savings-box .amount { font-size: 1.4rem; font-weight: 700; color: #86efac; }
  .savings-box .sub { font-size: 0.8rem; color: #4ade80; margin-top: 0.2rem; }

  @media (max-width: 600px) {
    .form-grid { grid-template-columns: 1fr; }
    .stats-grid { grid-template-columns: 1fr 1fr; }
  }
</style>
</head>
<body>
<div class="container">
  <h1>LLM Cost Calculator</h1>
  <p class="subtitle">Estimate monthly API costs across models — built for consultants and technical buyers.</p>

  <!-- ── INPUT FORM ── -->
  <div class="card">
    <h2>Your Use Case</h2>
    <form method="POST" action="/">
      <div class="form-grid">

        <div class="form-group full">
          <label for="use_case">Use case description</label>
          <input type="text" id="use_case" name="use_case"
                 placeholder="e.g. Customer support chatbot"
                 value="{{ form.use_case }}">
        </div>

        <div class="form-group">
          <label for="monthly_requests">Monthly requests</label>
          <input type="number" id="monthly_requests" name="monthly_requests"
                 min="1" placeholder="50000"
                 value="{{ form.monthly_requests }}" required>
        </div>

        <div class="form-group">
          <label for="avg_input">Avg input tokens / request</label>
          <input type="number" id="avg_input" name="avg_input"
                 min="1" placeholder="500"
                 value="{{ form.avg_input }}" required>
          <span class="hint">1,000 tokens ≈ 750 words</span>
        </div>

        <div class="form-group">
          <label for="avg_output">Avg output tokens / request</label>
          <input type="number" id="avg_output" name="avg_output"
                 min="1" placeholder="200"
                 value="{{ form.avg_output }}" required>
        </div>

        <div class="form-group">
          <label for="model">Model to evaluate</label>
          <select id="model" name="model">
            {% for key, m in models.items() %}
            <option value="{{ key }}" {% if form.model == key %}selected{% endif %}>
              {{ m.label }} — ${{ m.input }}/${{ m.output }} per 1M
            </option>
            {% endfor %}
          </select>
        </div>

      </div>

      <!-- Prompt caching section -->
      <div style="margin-top:1.25rem;">
        <div class="checkbox-row">
          <input type="checkbox" id="use_caching" name="use_caching"
                 {% if form.use_caching %}checked{% endif %}
                 onchange="document.getElementById('caching-extra').style.display=this.checked?'block':'none'">
          <label for="use_caching">System prompt is repeated across calls (prompt caching)</label>
        </div>
        <div id="caching-extra" style="{% if form.use_caching %}display:block{% endif %}">
          <div class="form-group">
            <label for="cached_tokens">System prompt tokens (cached)</label>
            <input type="number" id="cached_tokens" name="cached_tokens"
                   min="0" placeholder="800"
                   value="{{ form.cached_tokens }}">
            <span class="hint">Anthropic models only — 90% discount on cached tokens</span>
          </div>
        </div>
      </div>

      <button type="submit">Calculate Cost →</button>
    </form>
  </div>

  <!-- ── RESULTS ── -->
  {% if results %}
  {% set r = results.selected %}

  <!-- Context window warning -->
  {% if results.warn_context %}
  <div class="warning">
    ⚠️ Your avg input tokens ({{ form.avg_input | int | format_num }}) exceed 50,000. Verify the model's context
    window supports this before committing to a cost estimate.
  </div>
  {% endif %}

  <!-- Key metrics -->
  <div class="card">
    <h2>Results — {{ models[form.model].label }}{% if form.use_case %} · {{ form.use_case }}{% endif %}</h2>
    <div class="stats-grid">
      <div class="stat-box">
        <div class="value">{{ r.monthly | fmt_usd }}</div>
        <div class="label">Monthly cost</div>
      </div>
      <div class="stat-box">
        <div class="value">{{ r.annual | fmt_usd }}</div>
        <div class="label">Annual cost</div>
      </div>
      <div class="stat-box">
        <div class="value">{{ r.cost_per_req | fmt_usd_micro }}</div>
        <div class="label">Cost per request</div>
      </div>
    </div>

    <!-- Input vs output breakdown bar -->
    {% set total_cost = r.input_cost + r.output_cost %}
    {% if total_cost > 0 %}
    {% set in_pct  = (r.input_cost  / total_cost * 100) | round(1) %}
    {% set out_pct = (r.output_cost / total_cost * 100) | round(1) %}
    <div class="breakdown">
      <div class="bar-wrap">
        <div class="bar-input"  style="width:{{ in_pct }}%"></div>
        <div class="bar-output" style="width:{{ out_pct }}%"></div>
      </div>
      <div class="bar-legend">
        <span><span class="dot dot-input"></span>Input {{ in_pct }}% ({{ r.input_cost | fmt_usd }}/mo)</span>
        <span><span class="dot dot-output"></span>Output {{ out_pct }}% ({{ r.output_cost | fmt_usd }}/mo)</span>
      </div>
    </div>
    {% endif %}
  </div>

  <!-- Prompt caching savings -->
  {% if form.use_caching and results.savings %}
  <div class="card">
    <h2>Prompt Caching Savings (Anthropic Models)</h2>
    <div class="savings-box">
      <div class="title">Estimated monthly savings with caching enabled</div>
      <div class="amount">{{ results.savings.monthly_saved | fmt_usd }}</div>
      <div class="sub">
        Without caching: {{ results.savings.without | fmt_usd }}/mo →
        With caching: {{ results.savings.with_cache | fmt_usd }}/mo
        ({{ results.savings.pct | round(1) }}% reduction)
      </div>
    </div>
  </div>
  {% endif %}

  <!-- All-model comparison table -->
  <div class="card">
    <h2>Model Comparison (sorted by monthly cost)</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>$/1M in</th>
          <th>$/1M out</th>
          <th>Cost/req</th>
          <th>Monthly</th>
          <th>Annual</th>
        </tr>
      </thead>
      <tbody>
        {% for row in results.comparison %}
        <tr {% if row.model_key == form.model %}class="selected"{% endif %}>
          <td>
            {{ row.label }}
            <span class="badge badge-{{ row.provider }}">{{ row.provider }}</span>
            {% if row.model_key == form.model %}<span class="badge" style="background:#1e293b;color:#94a3b8;">selected</span>{% endif %}
          </td>
          <td>${{ row.in_price }}</td>
          <td>${{ row.out_price }}</td>
          <td>{{ row.cost_per_req | fmt_usd_micro }}</td>
          <td>{{ row.monthly | fmt_usd }}</td>
          <td>{{ row.annual  | fmt_usd }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Custom Jinja2 filters
# ---------------------------------------------------------------------------

@app.template_filter("fmt_usd")
def filter_fmt_usd(value):
    return fmt_usd(value)

@app.template_filter("fmt_usd_micro")
def filter_fmt_usd_micro(value):
    """For per-request costs that may be tiny fractions of a cent."""
    if value == 0:
        return "$0.000000"
    if value >= 0.01:
        return f"${value:.4f}"
    return f"${value:.6f}"

@app.template_filter("format_num")
def filter_format_num(value):
    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    # Preserve form values across POST for UX
    form = {
        "use_case":         "",
        "monthly_requests": "",
        "avg_input":        "",
        "avg_output":       "",
        "model":            "claude-sonnet-4",
        "use_caching":      False,
        "cached_tokens":    "",
    }
    results = None

    if request.method == "POST":
        # Read form inputs
        form["use_case"]         = request.form.get("use_case", "").strip()
        form["monthly_requests"] = request.form.get("monthly_requests", "0")
        form["avg_input"]        = request.form.get("avg_input", "0")
        form["avg_output"]       = request.form.get("avg_output", "0")
        form["model"]            = request.form.get("model", "claude-sonnet-4")
        form["use_caching"]      = "use_caching" in request.form
        form["cached_tokens"]    = request.form.get("cached_tokens", "0")

        # Parse numbers safely
        try:
            monthly_requests = int(form["monthly_requests"])
            avg_input        = int(form["avg_input"])
            avg_output       = int(form["avg_output"])
            cached_tokens    = int(form["cached_tokens"]) if form["cached_tokens"] else 0
        except ValueError:
            monthly_requests = avg_input = avg_output = cached_tokens = 0

        # Calculate cost for the selected model
        selected = calc_cost(
            form["model"], monthly_requests, avg_input, avg_output,
            cached_tokens=cached_tokens, use_caching=form["use_caching"]
        )

        # Calculate comparison table for all models (no caching applied to others)
        comparison = sorted(
            [calc_cost(k, monthly_requests, avg_input, avg_output) for k in MODELS],
            key=lambda x: x["monthly"]
        )

        # Prompt caching savings: compare Anthropic model with vs without caching
        savings = None
        if form["use_caching"] and cached_tokens > 0 and MODELS[form["model"]]["provider"] == "anthropic":
            without_cache = calc_cost(form["model"], monthly_requests, avg_input, avg_output, use_caching=False)
            with_cache    = calc_cost(form["model"], monthly_requests, avg_input, avg_output,
                                      cached_tokens=cached_tokens, use_caching=True)
            saved = without_cache["monthly"] - with_cache["monthly"]
            pct   = (saved / without_cache["monthly"] * 100) if without_cache["monthly"] > 0 else 0
            savings = {
                "monthly_saved": saved,
                "without":       without_cache["monthly"],
                "with_cache":    with_cache["monthly"],
                "pct":           pct,
            }

        results = {
            "selected":     selected,
            "comparison":   comparison,
            "savings":      savings,
            "warn_context": avg_input > 50_000,
        }

    return render_template_string(
        TEMPLATE,
        form=form,
        models=MODELS,
        results=results,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("LLM Cost Calculator running at http://localhost:5002")
    app.run(debug=True, port=5002)
