const express = require('express');
const cors = require('cors');
const fs = require('fs-extra');
const path = require('path');
const dayjs = require('dayjs');

const app = express();
const PORT = process.env.PORT || 54789; // Random port unlikely to be used

// HDR directory
const HDR_DIR = path.join(require('os').homedir(), '.hdr');
const CONFIG_FILE = path.join(HDR_DIR, 'config.json');
const LOG_FILE = path.join(HDR_DIR, 'llm_logs.jsonl');

// Ensure HDR directory exists
fs.ensureDirSync(HDR_DIR);

app.use(cors());
app.use(express.json());

// Serve static files from frontend build
app.use(express.static(path.join(__dirname, '../frontend/dist')));

// API Endpoints

// Get config
app.get('/api/config', async (req, res) => {
  try {
    let config = {};
    if (await fs.pathExists(CONFIG_FILE)) {
      config = await fs.readJson(CONFIG_FILE);
    }
    res.json(config);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Save config
app.post('/api/config', async (req, res) => {
  try {
    const config = req.body;
    await fs.writeJson(CONFIG_FILE, config, { spaces: 2 });
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get logs aggregated by month
app.get('/api/logs/month/:year/:month', async (req, res) => {
  try {
    const { year, month } = req.params;
    const page = parseInt(req.query.page || 1);
    const perPage = parseInt(req.query.perPage || 100);

    if (!(await fs.pathExists(LOG_FILE))) {
      return res.json({
        aggregated: [],
        logs: [],
        total: 0,
        page,
        perPage
      });
    }

    const logContent = await fs.readFile(LOG_FILE, 'utf8');
    const lines = logContent.trim().split('\n').filter(line => line);
    const allLogs = lines.map(line => JSON.parse(line)).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    // Filter logs for the requested month
    const monthStart = dayjs(`${year}-${month}-01`).startOf('month');
    const monthEnd = monthStart.endOf('month');

    const monthLogs = allLogs.filter(log => {
      const logDate = dayjs(log.timestamp);
      return logDate.isAfter(monthStart) && logDate.isBefore(monthEnd);
    });

    // Aggregate by day
    const aggregated = {};
    monthLogs.forEach(log => {
      const day = dayjs(log.timestamp).format('YYYY-MM-DD');
      if (!aggregated[day]) {
        aggregated[day] = {
          date: day,
          calls: 0,
          cached_input_tokens: 0,
          new_input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0
        };
      }
      aggregated[day].calls += 1;
      aggregated[day].cached_input_tokens += log.cached_input_tokens || 0;
      aggregated[day].new_input_tokens += log.new_input_tokens || 0;
      aggregated[day].output_tokens += log.output_tokens || 0;
      aggregated[day].total_tokens += log.total_tokens || 0;
    });

    // Pagination for logs
    const startIndex = (page - 1) * perPage;
    const endIndex = startIndex + perPage;
    const paginatedLogs = monthLogs.slice(startIndex, endIndex);

    res.json({
      aggregated: Object.values(aggregated).sort((a, b) => a.date.localeCompare(b.date)),
      logs: paginatedLogs,
      total: monthLogs.length,
      page,
      perPage
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get logs for a specific day
app.get('/api/logs/day/:date', async (req, res) => {
  try {
    const { date } = req.params;
    const page = parseInt(req.query.page || 1);
    const perPage = parseInt(req.query.perPage || 100);

    if (!(await fs.pathExists(LOG_FILE))) {
      return res.json({
        logs: [],
        total: 0,
        page,
        perPage
      });
    }

    const logContent = await fs.readFile(LOG_FILE, 'utf8');
    const lines = logContent.trim().split('\n').filter(line => line);
    const allLogs = lines.map(line => JSON.parse(line)).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    // Filter logs for the requested day
    const dayStart = dayjs(date).startOf('day');
    const dayEnd = dayStart.endOf('day');

    const dayLogs = allLogs.filter(log => {
      const logDate = dayjs(log.timestamp);
      return logDate.isAfter(dayStart) && logDate.isBefore(dayEnd);
    });

    // Pagination
    const startIndex = (page - 1) * perPage;
    const endIndex = startIndex + perPage;
    const paginatedLogs = dayLogs.slice(startIndex, endIndex);

    res.json({
      logs: paginatedLogs,
      total: dayLogs.length,
      page,
      perPage
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Catch all route for SPA
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/dist/index.html'));
});

app.listen(PORT, () => {
  console.log(`HDR WebUI running on http://localhost:${PORT}`);
});

module.exports = app;