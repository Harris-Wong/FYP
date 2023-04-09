const { spawn } = require('child_process');
const { apiFetch } = require('./data_fetch.js');

function predictionTask() {
  // Run 'Fetch Data' task
  console.log('Task "Fetch Data" process running...');
  const fetchDataProcess = spawn('node', ['-e', `${apiFetch.toString()}; apiFetch()`]);
  fetchDataProcess.stdout.on('data', (data) => {
    console.log(data.toString());
  });
  fetchDataProcess.stderr.on('data', (data) => {
    console.log(data.toString());
  });
  fetchDataProcess.on('close', (code) => {
    console.log(`Task 'Fetch Data' process exited with code ${code}`);
  });

  // Define the list of Python tasks
  const pythonTasks = [
    // 'task1.py'
  ]

  // Run Python task in sequence
  let i = 0;
  function runPythonTask() {
    if (i < pythonTasks.length) {
      const taskProcess = spawn('python', [pythonTasks[i]]);
      taskProcess.stdout.on('data', (data) => {
        console.log(data.toString());
      });
      taskProcess.stderr.on('data', (data) => {
        console.log(data.toString());
      });
      taskProcess.on('close', (code) => {
        console.log(`Task ${i+1} process exited with code $(code)`);
        i++;
        runPythonTask();
      });
    }
  }

  runPythonTask();
}

module.exports = { predictionTask };
