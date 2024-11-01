document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    const autoCompleteList = document.getElementById('autoComplete_list');

    searchInput.addEventListener('input', function() {
        const query = this.value.trim();
        if (query.length > 0) {
            fetch(`/movie_titles?term=${encodeURIComponent(query)}`)
                .then(response => response.json())
                .then(data => {
                    autoCompleteList.innerHTML = '';
                    if (data.length > 0) {
                        data.forEach(title => {
                            const li = document.createElement('li');
                            li.textContent = title;
                            li.addEventListener('click', function() {
                                searchInput.value = this.textContent;
                                autoCompleteList.innerHTML = '';
                            });
                            autoCompleteList.appendChild(li);
                        });
                        autoCompleteList.style.display = 'block';
                    } else {
                        const li = document.createElement('li');
                        li.textContent = 'No results found';
                        li.classList.add('no_result');
                        autoCompleteList.appendChild(li);
                        autoCompleteList.style.display = 'block';
                    }
                })
                .catch(error => {
                    console.error('Error fetching autocomplete data:', error);
                });
        } else {
            autoCompleteList.innerHTML = '';
            autoCompleteList.style.display = 'none';
        }
    });

    // Close the autocomplete list when clicking outside
    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !autoCompleteList.contains(e.target)) {
            autoCompleteList.innerHTML = '';
            autoCompleteList.style.display = 'none';
        }
    });
});
