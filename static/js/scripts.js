document.addEventListener('DOMContentLoaded', function() {
    const registerForm = document.getElementById('registerForm');
    const predictForm = document.getElementById('predictForm');

    registerForm.addEventListener('submit', function(event) {
        event.preventDefault();
        // ... (Logic to handle form submission for grid registration)
    });

    predictForm.addEventListener('submit', function(event) {
        event.preventDefault();
        // ... (Logic to handle form submission for predictions) 
    });
});