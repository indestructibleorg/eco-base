//# @GL-governed
//# @GL-layer: GL90-99
//# @GL-semantic: archive-tools
//# @GL-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
//#
//# GL Unified Charter Activated

import React, { useState } from 'react';
import axios from 'axios';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

// ============================================================================
// Constants
// ============================================================================

const INITIAL_TASKS = [
  { id: 'task-1', type: 'classical', config: { operation: 'preprocess', data: [] } },
  { id: 'task-2', type: 'quantum', config: { circuit: 'simple_x', shots: 100, backend: 'cirq' } },
];

const BACKEND_OPTIONS = ['cirq', 'qiskit', 'pennylane'];

// ============================================================================
// Sub-components
// ============================================================================

/**
 * ClassicalTaskConfig - Configuration form for classical tasks
 */
function ClassicalTaskConfig({ config, onUpdate }) {
  return (
    <div className="mt-2">
      <label className="block text-sm text-gray-600">Operation</label>
      <input
        type="text"
        value={config.operation}
        onChange={(e) => onUpdate('operation', e.target.value)}
        className="mt-1 block w-full p-2 border border-gray-300 rounded-md"
        placeholder="e.g., preprocess"
      />
    </div>
  );
}

/**
 * QuantumTaskConfig - Configuration form for quantum tasks
 */
function QuantumTaskConfig({ config, onUpdate }) {
  return (
    <div className="mt-2 space-y-2">
      <div>
        <label className="block text-sm text-gray-600">Circuit</label>
        <input
          type="text"
          value={config.circuit}
          onChange={(e) => onUpdate('circuit', e.target.value)}
          className="mt-1 block w-full p-2 border border-gray-300 rounded-md"
          placeholder="e.g., simple_x"
        />
      </div>
      <div>
        <label className="block text-sm text-gray-600">Shots</label>
        <input
          type="number"
          value={config.shots}
          onChange={(e) => onUpdate('shots', parseInt(e.target.value, 10))}
          className="mt-1 block w-full p-2 border border-gray-300 rounded-md"
          placeholder="e.g., 100"
        />
      </div>
      <div>
        <label className="block text-sm text-gray-600">Backend</label>
        <select
          value={config.backend}
          onChange={(e) => onUpdate('backend', e.target.value)}
          className="mt-1 block w-full p-2 border border-gray-300 rounded-md"
        >
          {BACKEND_OPTIONS.map((backend) => (
            <option key={backend} value={backend}>
              {backend.charAt(0).toUpperCase() + backend.slice(1)}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

/**
 * TaskCard - A draggable task card
 */
function TaskCard({ task, index, onUpdateConfig }) {
  const taskTypeLabel = task.type === 'classical' ? 'Classical Task' : 'Quantum Task';

  const handleConfigUpdate = (field, value) => {
    onUpdateConfig(task.id, field, value);
  };

  return (
    <Draggable draggableId={task.id} index={index}>
      {(provided) => (
        <div
          ref={provided.innerRef}
          {...provided.draggableProps}
          {...provided.dragHandleProps}
          className="p-4 bg-gray-50 rounded-md shadow-sm border border-gray-200"
        >
          <h3 className="text-lg font-semibold text-gray-800">
            {taskTypeLabel} #{index + 1}
          </h3>
          {task.type === 'classical' ? (
            <ClassicalTaskConfig config={task.config} onUpdate={handleConfigUpdate} />
          ) : (
            <QuantumTaskConfig config={task.config} onUpdate={handleConfigUpdate} />
          )}
        </div>
      )}
    </Draggable>
  );
}

/**
 * TaskList - Droppable task list container
 */
function TaskList({ tasks, onUpdateConfig }) {
  return (
    <Droppable droppableId="tasks">
      {(provided) => (
        <div {...provided.droppableProps} ref={provided.innerRef} className="space-y-4">
          {tasks.map((task, index) => (
            <TaskCard
              key={task.id}
              task={task}
              index={index}
              onUpdateConfig={onUpdateConfig}
            />
          ))}
          {provided.placeholder}
        </div>
      )}
    </Droppable>
  );
}

/**
 * StatusMessage - Displays error or success messages
 */
function StatusMessage({ error, success }) {
  return (
    <>
      {error && <div className="mb-4 text-red-600">{error}</div>}
      {success && <div className="mb-4 text-green-600">{success}</div>}
    </>
  );
}

// ============================================================================
// Helper functions
// ============================================================================

/**
 * Creates a new task with default configuration
 */
function createNewTask(taskCount, type) {
  const defaultConfig =
    type === 'classical'
      ? { operation: 'preprocess', data: [] }
      : { circuit: 'simple_x', shots: 100, backend: 'cirq' };

  return {
    id: `task-${taskCount + 1}`,
    type,
    config: defaultConfig,
  };
}

/**
 * Reorders tasks after drag and drop
 */
function reorderTasks(tasks, sourceIndex, destinationIndex) {
  const reorderedTasks = Array.from(tasks);
  const [movedTask] = reorderedTasks.splice(sourceIndex, 1);
  reorderedTasks.splice(destinationIndex, 0, movedTask);
  return reorderedTasks;
}

// ============================================================================
// Main component
// ============================================================================

function WorkflowDesigner() {
  const [workflowName, setWorkflowName] = useState('');
  const [tasks, setTasks] = useState(INITIAL_TASKS);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleDragEnd = (result) => {
    if (!result.destination) return;
    setTasks(reorderTasks(tasks, result.source.index, result.destination.index));
  };

  const handleAddTask = (type) => {
    const newTask = createNewTask(tasks.length, type);
    setTasks([...tasks, newTask]);
  };

  const handleUpdateTaskConfig = (id, field, value) => {
    setTasks(
      tasks.map((task) =>
        task.id === id ? { ...task, config: { ...task.config, [field]: value } } : task
      )
    );
  };

  const handleSubmit = async () => {
    if (!workflowName) {
      setError('Workflow name is required');
      return;
    }

    try {
      const response = await axios.post('/api/workflows', { name: workflowName, tasks });
      setSuccess(`Workflow '${workflowName}' created with ID: ${response.data.workflow_id}`);
      setError(null);
      setWorkflowName('');
      setTasks(INITIAL_TASKS);
    } catch (err) {
      setError(`Failed to create workflow: ${err.response?.data?.detail || err.message}`);
      setSuccess(null);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Workflow Designer</h2>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700">Workflow Name</label>
        <input
          type="text"
          value={workflowName}
          onChange={(e) => setWorkflowName(e.target.value)}
          className="mt-1 block w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          placeholder="Enter workflow name"
        />
      </div>

      <div className="mb-4 flex space-x-4">
        <button
          onClick={() => handleAddTask('classical')}
          className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition duration-300"
        >
          Add Classical Task
        </button>
        <button
          onClick={() => handleAddTask('quantum')}
          className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 transition duration-300"
        >
          Add Quantum Task
        </button>
      </div>

      <StatusMessage error={error} success={success} />

      <DragDropContext onDragEnd={handleDragEnd}>
        <TaskList tasks={tasks} onUpdateConfig={handleUpdateTaskConfig} />
      </DragDropContext>

      <button
        onClick={handleSubmit}
        className="mt-6 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-300"
      >
        Save Workflow
      </button>
    </div>
  );
}

export default WorkflowDesigner;
