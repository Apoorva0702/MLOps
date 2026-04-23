const articleInput = document.getElementById('article-input');
const predictBtn = document.getElementById('predict-btn');
const btnText = predictBtn.querySelector('.btn-text');
const spinner = predictBtn.querySelector('.spinner');

const resultsSection = document.getElementById('results-section');
const resultCard = document.querySelector('.result-card');
const predictionLabel = document.getElementById('prediction-label');
const confidenceScore = document.getElementById('confidence-score');

const feedbackSection = document.getElementById('feedback-section');
const feedbackThanks = document.getElementById('feedback-thanks');
const btnCorrect = document.getElementById('feedback-correct');
const btnIncorrect = document.getElementById('feedback-incorrect');

let currentPredictionId = null;
let currentLabel = null;

predictBtn.addEventListener('click', async () => {
    const text = articleInput.value.trim();
    if (!text) {
        alert("Please enter an article to analyze.");
        return;
    }

    // UI Loading state
    predictBtn.disabled = true;
    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    feedbackThanks.classList.add('hidden');
    feedbackSection.classList.remove('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const data = await response.json();
        
        // Update DOM
        currentPredictionId = data.id;
        currentLabel = data.prediction;
        
        predictionLabel.textContent = data.prediction;
        confidenceScore.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        
        // Styling classes
        resultCard.classList.remove('real', 'fake');
        if (data.prediction === 'REAL') {
            resultCard.classList.add('real');
        } else {
            resultCard.classList.add('fake');
        }

        // Show results
        resultsSection.classList.remove('hidden');

        // Scroll to results smoothly
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (error) {
        console.error("Error predicting:", error);
        alert("An error occurred while analyzing the text.");
    } finally {
        // Reset loading state
        predictBtn.disabled = false;
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
    }
});

async function submitFeedback(correctLabel) {
    if (!currentPredictionId) return;
    
    // Disable buttons
    btnCorrect.disabled = true;
    btnIncorrect.disabled = true;

    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                id: currentPredictionId,
                correct_label: correctLabel
            })
        });

        if (response.ok) {
            feedbackSection.classList.add('hidden');
            feedbackThanks.classList.remove('hidden');
        }
    } catch (error) {
        console.error("Error submitting feedback:", error);
    } finally {
        btnCorrect.disabled = false;
        btnIncorrect.disabled = false;
    }
}

btnCorrect.addEventListener('click', () => {
    // If it's correct, the "correct_label" is the one it predicted
    submitFeedback(currentLabel);
});

btnIncorrect.addEventListener('click', () => {
    // If it's incorrect, the "correct_label" is the opposite
    const oppositeLabel = document.getElementById('prediction-label').textContent === 'REAL' ? 'FAKE' : 'REAL';
    submitFeedback(oppositeLabel);
});
