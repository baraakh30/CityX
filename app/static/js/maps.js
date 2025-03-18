        // District data
        const districts = [
            "BAYVIEW", "CENTRAL", "INGLESIDE", "MISSION", "NORTHERN", 
            "PARK", "RICHMOND", "SOUTHERN", "TARAVAL", "TENDERLOIN"
        ];
        
        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {
            // Add smooth appearance for elements
            const elements = document.querySelectorAll('.content-wrapper, .map-selector, .map-container');
            
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
            
            // Initialize date range
            const dateInput = document.getElementById('searchDate');
            const today = new Date();
            const minDate = new Date("2003-01-06");
            const maxDate = new Date("2015-05-13");

            // If today's date is outside the range, set it to the minDate
            if (today < minDate) {
                dateInput.valueAsDate = minDate;
            } else if (today > maxDate) {
                dateInput.valueAsDate = maxDate;
            } else {
                dateInput.valueAsDate = today;
            }            
            // Populate districts
            populateDistricts();
            
            // Populate hours
            populateHours();
            
            // Set up the search form
            document.getElementById('crimeSearchForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Collect and validate form data
                const date = document.getElementById('searchDate').value;
                const selectedDistrictElements = document.querySelectorAll('#districtOptions .district-option.selected');
                const selectedDistricts = Array.from(selectedDistrictElements).map(el => el.getAttribute('data-district'));
                const selectedHourElements = document.querySelectorAll('#hourOptions .hour-option.selected');
                const selectedHours = Array.from(selectedHourElements).map(el => el.getAttribute('data-hour'));
                
                // Update hidden inputs
                document.getElementById('selectedDistricts').value = selectedDistricts.join(',');
                document.getElementById('selectedHours').value = selectedHours.join(',');
                
                // Validate
                if (!date) {
                    alert("Please select a date");
                    return;
                }
                
                if (selectedDistricts.length === 0) {
                    alert("Please select at least one district");
                    return;
                }
                
                if (selectedHours.length === 0) {
                    // If no hours selected, select all
                    selectAllHours();
                    document.getElementById('selectedHours').value = Array.from(
                        document.querySelectorAll('#hourOptions .hour-option.selected')
                    ).map(el => el.getAttribute('data-hour')).join(',');
                }
                
                // Generate the query parameters
                const params = new URLSearchParams();
                params.append('date', date);
                params.append('districts', selectedDistricts.join(','));
                params.append('hours', selectedHours.join(','));
                
                console.log(params.toString());
                // Show loading
                document.querySelector('.map-loader').style.display = 'flex';
                document.getElementById('map-frame').style.opacity = '0';
                
                // Update iframe src to load the search results
                document.getElementById('map-frame').src = `/crime-search-map?${params.toString()}`;
                
                // Update buttons
                const buttons = document.querySelectorAll('.btn');
                buttons.forEach(button => button.classList.remove('active'));
                
                // Close the search form
                closeSearchForm();
            });
        });
        
        function populateDistricts() {
            const container = document.getElementById('districtOptions');
            districts.forEach(district => {
                const element = document.createElement('div');
                element.className = 'district-option';
                element.setAttribute('data-district', district);
                element.textContent = district;
                element.onclick = function() {
                    this.classList.toggle('selected');
                };
                container.appendChild(element);
            });
        }
        
        function populateHours() {
            const container = document.getElementById('hourOptions');
            for (let i = 0; i < 24; i++) {
                const element = document.createElement('div');
                element.className = 'hour-option';
                element.setAttribute('data-hour', i);
                element.textContent = i;
                element.onclick = function() {
                    this.classList.toggle('selected');
                };
                container.appendChild(element);
            }
        }
        
        function selectAllDistricts() {
            const options = document.querySelectorAll('#districtOptions .district-option');
            options.forEach(option => option.classList.add('selected'));
        }
        
        function clearAllDistricts() {
            const options = document.querySelectorAll('#districtOptions .district-option');
            options.forEach(option => option.classList.remove('selected'));
        }
        
        function selectAllHours() {
            const options = document.querySelectorAll('#hourOptions .hour-option');
            options.forEach(option => option.classList.add('selected'));
        }
        
        function clearAllHours() {
            const options = document.querySelectorAll('#hourOptions .hour-option');
            options.forEach(option => option.classList.remove('selected'));
        }
        
        function selectTimeRange(timeOfDay) {
            // First clear all
            clearAllHours();
            
            // Then select based on time of day
            const options = document.querySelectorAll('#hourOptions .hour-option');
            if (timeOfDay === 'morning') {
                options.forEach(option => {
                    const hour = parseInt(option.getAttribute('data-hour'));
                    if (hour >= 6 && hour <= 11) option.classList.add('selected');
                });
            } else if (timeOfDay === 'afternoon') {
                options.forEach(option => {
                    const hour = parseInt(option.getAttribute('data-hour'));
                    if (hour >= 12 && hour <= 17) option.classList.add('selected');
                });
            } else if (timeOfDay === 'evening') {
                options.forEach(option => {
                    const hour = parseInt(option.getAttribute('data-hour'));
                    if (hour >= 18 && hour <= 23) option.classList.add('selected');
                });
            } else if (timeOfDay === 'night') {
                options.forEach(option => {
                    const hour = parseInt(option.getAttribute('data-hour'));
                    if (hour >= 0 && hour <= 5) option.classList.add('selected');
                });
            }
        }
        
        function showSearchForm() {
            document.getElementById('searchModal').style.display = 'flex';
            setTimeout(() => {
                document.getElementById('searchForm').classList.add('active');
            }, 10);
        }
        
        function closeSearchForm() {
            document.getElementById('searchForm').classList.remove('active');
            setTimeout(() => {
                document.getElementById('searchModal').style.display = 'none';
            }, 300);
        }
        
        function changeMap(mapFile) {
            // Show loading indicator
            document.querySelector('.map-loader').style.display = 'flex';
            document.getElementById('map-frame').style.opacity = '0';
            
            // Change map source
            document.getElementById('map-frame').src = '/static/' + mapFile;
            
            // Update active button
            const buttons = document.querySelectorAll('.btn');
            buttons.forEach(button => button.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        function mapLoaded() {
            setTimeout(() => {
                document.getElementById('map-frame').style.opacity = '1';
                document.querySelector('.map-loader').style.display = 'none';
            }, 500);
        }