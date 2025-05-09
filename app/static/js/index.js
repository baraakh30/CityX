      // Add smooth appearance for elements as they enter viewport
      document.addEventListener("DOMContentLoaded", function() {
        const elements = document.querySelectorAll('.card, h2, .report-table-container');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1
        });
        
        elements.forEach(element => {
            observer.observe(element);
        });
        
        // Check if there are initial reports and show the section
        const initialReports = document.querySelectorAll('#reportsTable tbody tr');
        if (initialReports.length > 0) {
            document.getElementById('reports-section').classList.remove('hidden');
        }
    });
    
    // Text prediction form submission
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const description = document.getElementById('description').value.trim(); // Trim whitespace
        if (!description) {
            alert('Please enter a crime description');
            return;
        }
        
        
        // Show loading state on the button
        const button = this.querySelector('button');
        const originalText = button.innerHTML;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        button.disabled = true;
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `description=${encodeURIComponent(description)}`
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('category').textContent = data.category;
            document.getElementById('severity').textContent = data.severity;
            
            const resultDiv = document.getElementById('result');
            resultDiv.className = '';
            resultDiv.classList.add(`severity-${data.severity}`);
            
            // Hide the result first to restart animation
            resultDiv.style.display = 'none';
            
            // Force reflow
            void resultDiv.offsetWidth;
            
            // Show with animation
            resultDiv.style.display = 'block';
            
            // Reset button
            button.innerHTML = originalText;
            button.disabled = false;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while making the prediction');
            button.innerHTML = originalText;
            button.disabled = false;
        });
    });
    
    // File upload functionality
    const fileInput = document.getElementById('fileInput');
    const dropArea = document.getElementById('dropArea');
    const fileList = document.getElementById('fileList');
    const uploadButtonGroup = document.getElementById('uploadButtonGroup');
    const uploadButton = document.getElementById('uploadButton');
    const reportsTable = document.getElementById('reportsTable');
    const reportsSection = document.getElementById('reports-section');
    
    // Selected files
    let selectedFiles = [];
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    // Handle selected files
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    function handleFiles(files) {
        if (files.length > 0) {
            Array.from(files).forEach(file => {
                // Only accept PDF files
                if (file.type === 'application/pdf') {
                    // Check if file is already in the list
                    const isDuplicate = selectedFiles.some(f => 
                        f.name === file.name && 
                        f.size === file.size && 
                        f.lastModified === file.lastModified
                    );
                    
                    if (!isDuplicate) {
                        selectedFiles.push(file);
                        addFileToList(file);
                    }
                } else {
                    console.error('Not a PDF file:', file.name);
                    // Optionally show error to user
                }
            });
            
            // Show upload button if files are selected
            if (selectedFiles.length > 0) {
                uploadButtonGroup.classList.remove('hidden');
            }
        }
    }
    
    function addFileToList(file) {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.dataset.fileName = file.name;
        
        fileItem.innerHTML = `
            <div class="file-item-name">
                <i class="fas fa-file-pdf file-item-icon"></i>
                ${file.name}
            </div>
            <span class="file-item-status status-pending">Pending</span>
        `;
        
        fileList.appendChild(fileItem);
    }
    
    // Process files when upload button is clicked
    uploadButton.addEventListener('click', processFiles);
    
    function processFiles() {
        if (selectedFiles.length === 0) return;
        
        // Disable upload button during processing
        uploadButton.disabled = true;
        uploadButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        // Process each file
        let completedFiles = 0;
        
        selectedFiles.forEach((file, index) => {
            const fileItem = document.querySelector(`.file-item[data-file-name="${file.name}"]`);
            if (fileItem) {
                const statusLabel = fileItem.querySelector('.file-item-status');
                statusLabel.className = 'file-item-status status-processing';
                statusLabel.textContent = 'Processing...';
            }
            
            // Create form data for this file
            const formData = new FormData();
            formData.append('pdf_file', file);
            
            // Send the file to the server
            fetch('/upload_pdf', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Update file status
                const fileItem = document.querySelector(`.file-item[data-file-name="${file.name}"]`);
                if (fileItem) {
                    const statusLabel = fileItem.querySelector('.file-item-status');
                    statusLabel.className = 'file-item-status status-success';
                    statusLabel.textContent = 'Completed';
                }
                
                // Add the report to the table
                addReportToTable(data.report);
                
                // Show the reports section if hidden
                reportsSection.classList.remove('hidden');
                
                // Check if all files are processed
                completedFiles++;
                if (completedFiles === selectedFiles.length) {
                    finishProcessing();
                }
            })
            .catch(error => {
                console.error('Error processing file:', error);
                
                const fileItem = document.querySelector(`.file-item[data-file-name="${file.name}"]`);
                if (fileItem) {
                    const statusLabel = fileItem.querySelector('.file-item-status');
                    statusLabel.className = 'file-item-status status-error';
                    statusLabel.textContent = 'Error';
                }
                
                // Check if all files are processed
                completedFiles++;
                if (completedFiles === selectedFiles.length) {
                    finishProcessing();
                }
            });
        });
    }
    
    function finishProcessing() {
        // Reset upload button
        uploadButton.disabled = false;
        uploadButton.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Process Selected Files';
        
        // Clear the file input and selected files array
        fileInput.value = '';
        selectedFiles = [];
    }
    
    function addReportToTable(report) {
        const tableBody = reportsTable.querySelector('tbody');
        const row = document.createElement('tr');
        row.className = 'highlight-row';
        // Add cells for each field
        row.innerHTML = `
            <td>${report['Report Number'] || 'N/A'}</td>
            <td>${report['Date & Time'] || 'N/A'}</td>
            <td>${report['Category'] || 'N/A'}</td>
            <td>${report['Detailed Description'] || 'N/A'}</td>
            <td>${report['DayOfWeek'] || 'N/A'}</td>
            <td>${report['Reporting Officer'] || 'N/A'}</td>
            <td>${report['Incident Location'] || 'N/A'}</td>
            <td>${report['Latitude (Y)'] && report['Longitude (X)'] ? `(${report['Latitude (Y)']}, ${report['Longitude (X)']})` : 'N/A'}</td>
            <td>${report['Police District'] || 'N/A'}</td>
            <td>${report['Resolution'] || 'N/A'}</td>
            <td>${report['Suspect Description'] || 'N/A'}</td>
            <td>${report['Victim Information'] || 'N/A'}</td>
            <td>${report['Severity'] !== null && report['Severity'] !== undefined ? report['Severity'] : 'N/A'}</td>
        `;
        // Add to beginning of table
        if (tableBody.firstChild) {
            tableBody.insertBefore(row, tableBody.firstChild);
        } else {
            tableBody.appendChild(row);
        }
    }