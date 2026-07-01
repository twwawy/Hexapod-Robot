# Hardware Parts List

이 문서는 Hexapod Robot 제작에 사용한 주요 하드웨어 부품 목록을 정리한 문서입니다.

---

## 1. Main Controller / Computing Unit

| 제품명 | 개수 | 용도 | 구매링크 | 비고 |
|---|---:|---|---|---|
| Jetson Orin Nano Super | 1개 | 각종 센서를 종합해 이동 경로와 보행 방법 계산 | <https://www.devicemart.co.kr/goods/view?no=14990359> | - |
| STM32 NUCLEO-F446RE | 1개 | GPS, IMU, LoRa 통신 담당 | <https://www.devicemart.co.kr/goods/view?no=1376886> | - |
| SSD | 1개 | Jetson Orin Nano Super에 장착해 OS와 모델 구동 | <https://www.devicemart.co.kr/goods/view?no=12147824> | 제품명과 구매링크 확인 필요 |

---

## 2. Actuator / Joint Parts

| 제품명 | 개수 | 용도 | 구매링크 | 비고 |
|---|---:|---|---|---|
| DS51150-270 Servo Motor | 20개 | 로봇 다리 관절 구동용 서보모터 | AliExpress 상품 링크 확인 필요 | - |
| DS51150-270 Servo Horn 5pcs | 4개 | 서보모터와 로봇 관절 연결 | <https://ko.aliexpress.com/i/1005007393245886.html> | - |
| 6703ZZ Deep Groove Ball Bearing | 20개 | 다리 관절 회전 보조 | <https://www.coupang.com/vp/products/6143283810?itemId=11802532451&vendorItemId=79076107061&q=6703ZZ&searchId=d0d0392a5285022&sourceType=search&itemsCount=36&searchRank=0&rank=0&traceId=mls5mr3f> | - |

---

## 3. Mechanical Parts

| 제품명 | 개수 | 용도 | 구매링크 | 비고 |
|---|---:|---|---|---|
| PLA Filament | 4개 | 로봇 프레임 및 구조물 3D 프린팅 | <https://www.coupang.com/vp/products/6326227472?itemId=13209923756&vendorItemId=92097957520&q=pla+필라멘트&itemsCount=36&searchId=2c9f4d179589258&rank=3&searchRank=3&isAddedCart=https://www.coupang.com/vp/products/6326227472?itemId=13209923756&vendorItemId=92097957475&q=pla+필라멘트&itemsCount=36&searchId=2c9f4d179589258&rank=3&searchRank=3&isAddedCart=> | - |
| 스텐 육각렌치볼트 M4×10 | 1봉 | 로봇 조립용 볼트 | <https://www.coupang.com/vp/products/7877154060?utm_source=chatgpt.com> | - |

---

## 4. Power System

| 제품명 | 개수 | 용도 | 구매링크 | 비고 |
|---|---:|---|---|---|
| DXF Li-ion Battery 3S 7200mAh 100C | 2개 | 서보모터 전원 공급 | <https://rc9.co.kr/product/특가행사dxf-배터리-리튬-111v-7200mah-100c3s-dxf-한국총판-rc9-정품dxf01/80067/display/1/> | - |
| DXF Li-ion Battery 2S 7200mAh 100C | 1개 | 보드 및 센서 전원 공급 | <https://ko.aliexpress.com/item/1005008268788282.html?gatewayAdapt=glo2kor> | - |
| 고출력 스텝다운 5A 가변 DC 컨버터 | 2개 | 배터리 전압 강하용 DC-DC 컨버터 | <https://www.devicemart.co.kr/goods/view?no=1321141> | - |
| 니토 미니 배터리 잔량표시 볼트메타 | 3개 | 배터리 잔량 및 전압 확인 | <https://www.coupang.com/vp/products/334419473?itemId=1067530756&vendorItemId=75456141668&q=리튬배터리+3S+잔량표시&itemsCount=36&searchId=6dd464417351806&rank=2&searchRank=2&isAddedCart=> | - |
| 아두이노 3채널 5V 릴레이 모듈 | 2개 | 각 다리 서보모터 전원 공급 및 차단 | <https://www.devicemart.co.kr/goods/view?no=12147824> | 구매링크 확인 필요 |
| CANAL MR2-110-C5-BB Switch | 1개 | 로봇 메인 전원 스위치 | <https://www.devicemart.co.kr/goods/view?no=38183> | - |

---

## 5. Connector / Wiring

| 제품명 | 개수 | 용도 | 구매링크 | 비고 |
|---|---:|---|---|---|
| AMASS XT60E-F Female Connector | 3개 | 배터리 고정 및 충전용 커넥터 | <https://www.rcbank.co.kr/m2/goods/view.php?goodsno=95419> | - |
| JST-XH Balance Charge Wire | 3개 | 배터리 밸런스잭 연장선 | <https://hobbyzone.kr/product/jst-xh-2s3s4s6s-20cm-22awg-balance-charge-wire-배터리-밸런스잭-연장선/5304/> | 2S용 1개, 3S용 2개 |
| XT60-F 암/수 커넥터 | 3개 | 배터리-전선 연결 소켓 | <https://www.devicemart.co.kr/goods/view?no=1078005> | - |
| 10AWG 실리콘 케이블 | 10개 | 메인 전원용 굵은 전선 | <https://www.coupang.com/vp/products/8537462031?itemId=24715590149&searchId=8327ed906c3b463ea0b5a524e92736ec&sourceType=brandstore_sdp_atf-best_products&storeId=10240&subSourceType=brandstore_sdp_atf-best_products&vendorId=A00502821&vendorItemId=90750092729> | - |
| 18AWG 실리콘 케이블 | 10개 | 모터 전원용 전선 | <https://www.coupang.com/vp/products/8264903762?itemId=23813552001&searchId=6246d3a0cbd5455f8f25c7fffbb5633c&sourceType=brandstore_sdp_atf-best_products&storeId=10240&subSourceType=brandstore_sdp_atf-best_products&vendorId=A00502821&vendorItemId=90837339910> | - |
| 22AWG × 4C 전선 | 20개 | 센서 및 신호용 얇은 전선 | <https://www.coupang.com/vp/products/7156740267?itemId=18003337265&searchId=492e19c0b188424685e18897d2f7e036&sourceType=brandstore_sdp_atf-best_products&storeId=34612&subSourceType=brandstore_sdp_atf-best_products&vendorId=A00340725&vendorItemId=85159319229#sdpReview> | - |
| LiDAR Cable | 1개 | Livox Mid-360 LiDAR와 Jetson Orin Nano Super 간 통신 | 구매링크 확인 필요 | 제품명과 구매링크 확인 필요 |

---

## 6. Sensor / Perception

| 제품명 | 개수 | 용도 | 구매링크 | 비고 |
|---|---:|---|---|---|
| Livox Mid-360 LiDAR | 1개 | 주변 지형 및 장애물 인식 | <https://www.coupang.com/vp/products/9250130341?itemId=27358998182&vendorItemId=94325064888&src=1042503&spec=10304025&addtag=400&ctag=9250130341&lptag=9250130341-27358998182&itime=20260219004855&pageType=PRODUCT&pageValue=9250130341&wPcid=17511806024806833762516&wRef=www.google.com&wTime=20260219004855&redirect=landing&gclid=&mcid=9ea861c46b59402eb063721237189ad6&campaignid=&adgroupid=> | - |
| Intel RealSense D435 | 1개 | 주변 지형 및 깊이 정보 인식 | <https://www.devicemart.co.kr/goods/view?no=13017227> | - |
| WT931 9DOF IMU | 1개 | 로봇의 기울기와 방향 측정 | <https://www.devicemart.co.kr/goods/view?no=10886274> | - |
| Interlink FSR 402 Pressure Sensor | 6개 | 로봇 발의 지면 접촉 여부 확인 | <https://www.devicemart.co.kr/goods/view?no=33870> | - |
| ADC Module | 3개 | STM32의 부족한 ADC 채널 보완 | 제품명 및 구매링크 확인 필요 | 원본 목록에서 릴레이 모듈로 적혀 있어 확인 필요 |
| NEO-M8N GPS Module | 1개 | 로봇의 실시간 위치 좌표 확인 | <https://www.devicemart.co.kr/goods/view?no=15106909> | - |

---

## 7. Communication / Remote Control

| 제품명 | 개수 | 용도 | 구매링크 | 비고 |
|---|---:|---|---|---|
| RadioMaster Pocket ELRS 조종기 | 1개 | 로봇 수동 조종용 컨트롤러 | <https://susungrc.com/product/%EB%9D%BC%EB%94%94%EC%98%A4%EB%A7%88%EC%8A%A4%ED%84%B0-pocket-%EC%A1%B0%EC%A2%85%EA%B8%B0-elrs/2385/> | - |
| ELRS Receiver Module | 1개 | 조종기 신호를 수신하여 로봇 제어부로 전달 | <https://www.devicemart.co.kr/goods/view?no=15138136> | - |
| RYLR998 UART LoRa Module | 2개 | 로봇과 컴퓨터 간 장거리 무선 통신 | <https://www.coupang.com/vp/products/7232535124?itemId=18355967552&vendorItemId=85499819592&src=1042503&spec=10304025&addtag=400&ctag=7232535124&lptag=7232535124-18355967552&itime=20260228152610&pageType=PRODUCT&pageValue=7232535124&wPcid=17511806024806833762516&wRef=www.google.com&wTime=20260228152610&redirect=landing&gclid=CjwKCAiAnoXNBhAZEiwAnItcG2BsnsktZJzYX85kpdk3TqxUEqCcQUz-7MKTnwaD-J5tWiEd4ZkqYhoC9DkQAvD_BwE&mcid=c32dedb64c9944bcab7d012420c4f76a&campaignid=22815108882&adgroupid=> | - |
