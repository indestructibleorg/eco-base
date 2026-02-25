//# @GL-governed
//# @GL-layer: GL90-99
//# @GL-semantic: archive-tools
//# @GL-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
//#
//# GL Unified Charter Activated

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

// ============================================================================
// Sub-components
// ============================================================================

/**
 * WorkflowCard - Displays a single workflow status
 */
function WorkflowCard({ workflow }) {
  const statusColorClass = workflow.status === 'completed' ? 'text-green-600' : 'text-yellow-600';

  return (
    <div className="p-4 bg-gray-50 rounded-md shadow-sm border border-gray-200">
      <h4 className="text-lg font-medium text-gray-800">
        {workflow.name} (ID: {workflow.workflow_id})
      </h4>
      <p className="text-gray-600">
        Status: <span className={`font-semibold ${statusColorClass}`}>{workflow.status}</span>
      </p>
    </div>
  );
}

/**
 * WorkflowSection - Displays all workflows
 */
function WorkflowSection({ workflows }) {
  return (
    <div className="mb-8">
      <h3 className="text-xl font-semibold text-gray-700 mb-2">Workflow Status</h3>
      {workflows.length === 0 ? (
        <p className="text-gray-500">No workflows found.</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {workflows.map((workflow) => (
            <WorkflowCard key={workflow.workflow_id} workflow={workflow} />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * MetricCard - Displays a single metric
 */
function MetricCard({ metric }) {
  return (
    <div className="p-4 bg-gray-50 rounded-md shadow-sm border border-gray-200">
      <h4 className="text-lg font-medium text-gray-800">
        Workflow {metric.workflow_id} - Task {metric.task_id}
      </h4>
      <p className="text-gray-600">Runtime: {metric.runtime.toFixed(2)}s</p>
      {metric.circuit_depth && (
        <p className="text-gray-600">Circuit Depth: {metric.circuit_depth}</p>
      )}
      {metric.shots && <p className="text-gray-600">Shots: {metric.shots}</p>}
      <p className="text-gray-500 text-sm">Timestamp: {metric.timestamp}</p>
    </div>
  );
}

/**
 * MetricsSection - Displays performance metrics with chart
 */
function MetricsSection({ metrics, chartData, chartOptions }) {
  return (
    <div className="mb-8">
      <h3 className="text-xl font-semibold text-gray-700 mb-2">Performance Metrics</h3>
      {metrics.length === 0 ? (
        <p className="text-gray-500">No metrics available.</p>
      ) : (
        <div className="space-y-4">
          <div className="p-4 bg-gray-50 rounded-md shadow-sm border border-gray-200">
            <Line data={chartData} options={chartOptions} />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {metrics.map((metric) => (
              <MetricCard key={`${metric.workflow_id}-${metric.task_id}`} metric={metric} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Chart configuration
// ============================================================================

const CHART_OPTIONS = {
  responsive: true,
  plugins: {
    legend: {
      position: 'top',
    },
    title: {
      display: true,
      text: 'Workflow Performance Metrics',
    },
  },
  scales: {
    y: {
      beginAtZero: true,
      title: {
        display: true,
        text: 'Value',
      },
    },
  },
};

/**
 * Builds chart data from metrics
 */
function buildChartData(metrics) {
  return {
    labels: metrics.map((m) => `Task ${m.task_id}`),
    datasets: [
      {
        label: 'Runtime (seconds)',
        data: metrics.map((m) => m.runtime),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        tension: 0.1,
      },
      {
        label: 'Shots (quantum tasks)',
        data: metrics.map((m) => m.shots || 0),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.5)',
        tension: 0.1,
      },
    ],
  };
}

// ============================================================================
// Main component
// ============================================================================

/**
 * Safely fetches data from an API endpoint
 * Returns { data: response, error: null } on success
 * Returns { data: null, error: errorMessage } on failure
 */
async function safeFetch(url) {
  try {
    const response = await axios.get(url);
    return { data: response.data, error: null };
  } catch (err) {
    return { data: null, error: err.response?.data?.detail || err.message };
  }
}

function Dashboard() {
  const [workflows, setWorkflows] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      const [workflowResult, metricsResult] = await Promise.all([
        safeFetch('/api/workflows'),
        safeFetch('/api/performance'),
      ]);

      // Collect any errors
      const errors = [];
      if (workflowResult.error) errors.push(`Workflows: ${workflowResult.error}`);
      if (metricsResult.error) errors.push(`Metrics: ${metricsResult.error}`);

      // Set data even if partially available
      if (workflowResult.data) setWorkflows(workflowResult.data);
      if (metricsResult.data) setMetrics(metricsResult.data);

      // Report combined errors if any
      if (errors.length > 0) {
        setError(`Failed to fetch data: ${errors.join('; ')}`);
      }

      setLoading(false);
    };

    fetchData();
  }, []);

  const chartData = buildChartData(metrics);

  return (
    <div className="p-6 bg-white rounded-lg shadow-md max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">QuantumFlow Dashboard</h2>

      {loading && <div className="text-gray-600">Loading...</div>}
      {error && <div className="text-red-600 mb-4">{error}</div>}

      <WorkflowSection workflows={workflows} />
      <MetricsSection metrics={metrics} chartData={chartData} chartOptions={CHART_OPTIONS} />
    </div>
  );
}

export default Dashboard;
