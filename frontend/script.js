document.addEventListener('DOMContentLoaded', () => {
    // DOM 요소 가져오기
    const imageUpload = document.getElementById('imageUpload');
    const searchButton = document.getElementById('searchButton');
    const resultDiv = document.getElementById('result');
    const errorResultDiv = document.getElementById('errorResult');
    const spinner = document.getElementById('spinner');

    const queryImage = document.getElementById('queryImage');
    const queryCanvas = document.getElementById('queryCanvas');
    const resultVideo = document.getElementById('resultVideo');
    const resultCanvas = document.getElementById('resultCanvas');
    const resultInfo = document.getElementById('resultInfo');
    const mainResultTitle = document.getElementById('main-result-title');
    const resultList = document.getElementById('result-list');

    const API_URL = 'http://127.0.0.1:8000';
    let currentQueryFile = null;

    // 검색 버튼 클릭 이벤트
    searchButton.addEventListener('click', async () => {
        const file = imageUpload.files[0];
        if (!file) {
            alert('검색할 이미지를 선택해주세요.');
            return;
        }
        currentQueryFile = file;

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
                // 결과가 배열이고, 비어있지 않은지 확인
                if (Array.isArray(data.result) && data.result.length > 0) {
                    displayResults(data.result);
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
     * 검색 결과(배열)를 화면에 표시하는 함수
     * @param {Array} results - 백엔드에서 받은 결과 객체 배열
     */
    function displayResults(results) {
        resultDiv.style.display = 'block';
        errorResultDiv.style.display = 'none';

        // 1. 가장 정확한 첫 번째 결과를 메인 화면에 표시
        displayMainResult(results[0], 1);

        // 2. 전체 결과 리스트를 하단에 표시
        resultList.innerHTML = ''; // 이전 리스트 초기화
        results.forEach((result, index) => {
            const resultItem = document.createElement('div');
            resultItem.className = 'col-lg-3 col-md-4 col-6';
            
            resultItem.innerHTML = `
                <div class="card h-100 result-card" role="button" title="클릭하여 자세히 보기">
                    <img src="${result.matched_frame}" class="card-img-top" alt="Result ${index + 1}">
                    <div class="card-body p-2">
                        <p class="card-text small mb-0 text-truncate">${result.video_file}</p>
                        <p class="card-text small text-muted">Time: ${result.timestamp}s | Score: ${result.score}</p>
                    </div>
                </div>
            `;

            // 각 결과 아이템 클릭 시 메인 화면을 해당 결과로 변경
            resultItem.addEventListener('click', () => {
                displayMainResult(result, index + 1);
                resultDiv.scrollIntoView({ behavior: 'smooth' });
            });

            resultList.appendChild(resultItem);
        });
    }

    /**
     * 하나의 결과 객체를 받아 메인 디스플레이 영역을 업데이트하는 함수
     * @param {object} result - 결과 객체
     * @param {number} rank - 결과 순위
     */
    function displayMainResult(result, rank) {
        if (!currentQueryFile) return;

        // 1. 쿼리 이미지 표시
        const queryURL = URL.createObjectURL(currentQueryFile);
        queryImage.src = queryURL;

        // 2. 결과 정보 텍스트 업데이트
        const { video_file, timestamp, score } = result;
        mainResultTitle.textContent = `검색된 장면 (Top ${rank})`;
        resultInfo.innerHTML = `<strong>파일:</strong> ${video_file}<br><strong>시간:</strong> ${timestamp}초 (유사도 점수: ${score})`;
        
        // 3. 결과 비디오 로드 및 시간 이동
        resultVideo.src = `${API_URL}/videos/${video_file}`;
        resultVideo.onloadedmetadata = () => {
            resultVideo.currentTime = parseFloat(timestamp);
        };

        // 4. 이미지와 비디오가 로드된 후 캔버스에 그리기
        const { query_keypoints, frame_keypoints } = result;
        let isQueryImageLoaded = false;
        let isVideoLoaded = false;

        const drawWhenReady = () => {
            if (isQueryImageLoaded && isVideoLoaded) {
                drawMatches(queryImage, queryCanvas, resultVideo, resultCanvas, query_keypoints, frame_keypoints);
            }
        };

        queryImage.onload = () => {
            isQueryImageLoaded = true;
            drawWhenReady();
        };

        resultVideo.onloadeddata = () => {
            isVideoLoaded = true;
            drawWhenReady();
        };

        resultVideo.onplay = () => { resultCanvas.style.display = 'none'; };
        resultVideo.onpause = () => { resultCanvas.style.display = 'block'; };
    }

    function drawMatches(img1, canvas1, vid2, canvas2, points1, points2) {
        canvas1.width = img1.clientWidth;
        canvas1.height = img1.clientHeight;
        canvas2.width = vid2.clientWidth;
        canvas2.height = vid2.clientHeight;

        const ctx1 = canvas1.getContext('2d');
        const ctx2 = canvas2.getContext('2d');
        
        ctx1.clearRect(0,0,canvas1.width, canvas1.height);
        ctx2.clearRect(0,0,canvas2.width, canvas2.height);

        drawKeypoints(ctx1, points1, img1.naturalWidth, img1.clientWidth);
        drawKeypoints(ctx2, points2, vid2.videoWidth, vid2.clientWidth);
        
        canvas2.style.display = 'block'; 
    }

    function drawKeypoints(ctx, keypoints, originalMediaWidth, displayMediaWidth) {
        const scale = displayMediaWidth / originalMediaWidth;
        ctx.strokeStyle = 'lime';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'red';

        keypoints.forEach(p => {
            const x = p[0] * scale;
            const y = p[1] * scale;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    function showError(message) {
        resultDiv.style.display = 'none';
        errorResultDiv.style.display = 'block';
        errorResultDiv.innerHTML = `<strong>오류:</strong> ${message}`;
    }
});