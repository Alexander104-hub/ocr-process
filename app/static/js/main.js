document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const loading = document.getElementById('loading');
    const errorDiv = document.getElementById('error');

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        // Show loading spinner
        loading.classList.remove('d-none');
        errorDiv.classList.add('d-none');

        const formData = new FormData(form);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Redirect to results page
                window.location.href = data.redirect;
            } else {
                // Show error message
                loading.classList.add('d-none');
                errorDiv.textContent = data.error || 'An error occurred during processing';
                errorDiv.classList.remove('d-none');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loading.classList.add('d-none');
            errorDiv.textContent = 'An error occurred during the upload process';
            errorDiv.classList.remove('d-none');
        });
    });
});
