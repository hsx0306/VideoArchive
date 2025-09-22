document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const searchButton = document.getElementById('searchButton');
    const resultDiv = document.getElementById('result');
    const errorResultDiv = document.getElementById('errorResult');
    const spinner = document.getElementById('spinner');

    const queryImage = document.getElementById('queryImage');
    const resultVideo = document.getElementById('resultVideo');
    const resultInfo = document.getElementById('resultInfo');
    const mainResultTitle = document.getElementById('main-result-title');
    const resultList = document.getElementById('result-list');

    const API_URL = 'http://127.0.0.1:8000';

    searchButton.addEventListener('click', async () => {
        const file = imageUpload.files[0];
        if (!file) {
            alert('검색할 이미지를 선택해주세요.');
            return;
        }

        spinner.style.display = 'inline-block';
        searchButton.disabled = true;
        resultDiv.style.display = 'none';
        errorResultDiv.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_URL}/search`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok && data.success) {
                if (Array.isArray(data.result) && data.result.length > 0) {
                    displayResults(data.result, file);
                } else {
                    showError("유사한 장면을 찾지 못했습니다.");
                }
            } else {
                showError(data.detail || data.message || '알 수 없는 오류가 발생했습니다.');
            }

        } catch (error) {
            showError('네트워크 오류: 서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인하세요.');
            console.error('Error:', error);
        } finally {
            spinner.style.display = 'none';
            searchButton.disabled = false;
        }
    });

    /**
     * 검색 결과를 화면에 표시하는 함수
     * @param {Array} results - 백엔드에서 받은 결과 객체 배열
     * @param {File} queryFile - 사용자가 업로드한 이미지 파일
     */
    function displayResults(results, queryFile) {
        resultDiv.style.display = 'block';
        resultList.innerHTML = ''; // Clear previous similar scenes

        // 1. Display the top result in the main area
        const topResult = results[0];
        displayMainResult(topResult, queryFile);

        // 2. Display other similar scenes
        if (results.length > 1) {
            const otherResults = results.slice(1);
            otherResults.forEach((res, index) => {
                const videoUrl = `${API_URL}/videos/${res.video_id}#t=${res.timestamp}`;

                const card = document.createElement('div');
                card.className = 'col-md-4';
                card.innerHTML = `
                    <div class="card result-card">
                        <video class="card-img-top" controls muted loop src="${videoUrl}"></video>
                        <div class="card-body">
                            <h6 class="card-title">유사 장면 #${index + 2}</h6>
                            <p class="card-text mb-1"><strong>파일:</strong> ${res.video_id}</p>
                            <p class="card-text"><strong>시간:</strong> ${parseFloat(res.timestamp).toFixed(2)}s | <strong>점수:</strong> ${res.score}</p>
                        </div>
                    </div>
                `;
                resultList.appendChild(card);
            });
        }
    }

    /**
     * 메인 검색 결과를 표시하는 함수
     * @param {object} result - 최상위 결과 객체
     * @param {File} queryFile - 사용자가 업로드한 이미지 파일
     */
    function displayMainResult(result, queryFile) {
        // Display query image
        queryImage.src = URL.createObjectURL(queryFile);

        // Display result video
        const { video_id, timestamp, score } = result;
        mainResultTitle.textContent = '검색된 장면 (Top 1)';
        resultInfo.innerHTML = `<strong>파일:</strong> ${video_id}<br><strong>시간:</strong> ${parseFloat(timestamp).toFixed(2)}초 (유사도 점수: ${score})`;
        
        resultVideo.src = `${API_URL}/videos/${video_id}`;
        resultVideo.onloadedmetadata = () => {
            resultVideo.currentTime = parseFloat(timestamp);
        };
    }

    function showError(message) {
        resultDiv.style.display = 'none';
        errorResultDiv.style.display = 'block';
        errorResultDiv.innerHTML = `<strong>오류:</strong> ${message}`;
    }
});