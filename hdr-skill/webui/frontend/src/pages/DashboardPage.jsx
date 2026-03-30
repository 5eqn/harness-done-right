import { useState, useEffect } from 'react'
import axios from 'axios'
import dayjs from 'dayjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

export default function DashboardPage() {
  const [selectedMonth, setSelectedMonth] = useState(dayjs().format('YYYY-MM'))
  const [aggregatedData, setAggregatedData] = useState([])
  const [logs, setLogs] = useState([])
  const [selectedDay, setSelectedDay] = useState(null)
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const perPage = 100

  useEffect(() => {
    loadMonthData()
  }, [selectedMonth])

  useEffect(() => {
    if (selectedDay) {
      loadDayData()
    }
  }, [selectedDay, page])

  const loadMonthData = async () => {
    try {
      const [year, month] = selectedMonth.split('-')
      const res = await axios.get(`/api/logs/month/${year}/${month}`)
      setAggregatedData(res.data.aggregated)
      setLogs(res.data.logs)
      setTotal(res.data.total)
      setPage(1)
      setSelectedDay(null)
    } catch (err) {
      console.error('Failed to load month data:', err)
    }
  }

  const loadDayData = async () => {
    try {
      const res = await axios.get(`/api/logs/day/${selectedDay}?page=${page}&perPage=${perPage}`)
      setLogs(res.data.logs)
      setTotal(res.data.total)
    } catch (err) {
      console.error('Failed to load day data:', err)
    }
  }

  const handleMonthChange = (e) => {
    setSelectedMonth(e.target.value)
  }

  const handleDayClick = (date) => {
    setSelectedDay(date)
    setPage(1)
  }

  const chartData = {
    labels: aggregatedData.map(d => d.date),
    datasets: [
      {
        label: 'Total Tokens',
        data: aggregatedData.map(d => d.total_tokens),
        borderColor: 'rgb(99, 102, 241)',
        backgroundColor: 'rgba(99, 102, 241, 0.5)',
      },
    ],
  }

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Daily Token Consumption',
      },
    },
    onClick: (e, elements) => {
      if (elements.length > 0) {
        const index = elements[0].index
        const date = aggregatedData[index].date
        handleDayClick(date)
      }
    }
  }

  const totalPages = Math.ceil(total / perPage)

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <div className="flex items-center gap-4">
          <label className="text-sm font-medium text-gray-700">Month:</label>
          <input
            type="month"
            value={selectedMonth}
            onChange={handleMonthChange}
            className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          />
          {selectedDay && (
            <button
              onClick={() => {
                setSelectedDay(null)
                loadMonthData()
              }}
              className="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 rounded"
            >
              Back to Month View
            </button>
          )}
        </div>
      </div>

      {selectedDay && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded">
          <p className="text-blue-800">
            Showing logs for <strong>{selectedDay}</strong>
          </p>
        </div>
      )}

      {!selectedDay && aggregatedData.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
          <Line data={chartData} options={chartOptions} />
          <p className="mt-2 text-sm text-gray-500 text-center">
            Click on a data point to view details for that day
          </p>
        </div>
      )}

      {!selectedDay && aggregatedData.length === 0 && (
        <div className="bg-white p-6 rounded-lg shadow-sm mb-6 text-center text-gray-500">
          No data available for the selected month
        </div>
      )}

      <div className="bg-white rounded-lg shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">
            {selectedDay ? `Logs for ${selectedDay}` : 'Recent Logs'}
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            {total} entries, showing page {page} of {totalPages}
          </p>
        </div>

        {logs.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            No logs found
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Time
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Cached
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      New Input
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Output
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Total
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {logs.map((log, idx) => (
                    <tr key={idx} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {dayjs(log.timestamp).format('HH:mm:ss')}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {log.type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {log.cached ? '✅' : '❌'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {log.new_input_tokens || 0}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {log.output_tokens || 0}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {log.total_tokens || 0}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        {log.success ? (
                          <span className="text-green-600">Success</span>
                        ) : (
                          <span className="text-red-600">Failed</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="px-6 py-4 border-t border-gray-200 flex justify-between items-center">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                className="px-3 py-1 border border-gray-300 rounded disabled:opacity-50"
              >
                Previous
              </button>
              <span className="text-sm text-gray-700">
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="px-3 py-1 border border-gray-300 rounded disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
