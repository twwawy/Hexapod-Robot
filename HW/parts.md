# Hardware Parts List

이 문서는 Hexapod Robot 제작에 사용한 주요 하드웨어 부품 목록을 정리한 문서입니다.

---

## 1. Main Controller / Computing Unit

### Jetson Orin Nano Super

- **개수**: 1개
- **용도**: 각종 센서를 종합해 이동 경로와 보행 방법 계산
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=14990359)

### STM32 NUCLEO-F446RE

- **개수**: 1개
- **용도**: GPS, IMU, LoRa 통신 담당
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=1376886)

### SSD

- **개수**: 1개
- **용도**: Jetson Orin Nano Super에 장착해 OS와 모델 구동
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=12147824)

---

## 2. Actuator / Joint Parts

### DS51150-270 Servo Motor

- **개수**: 20개
- **용도**: 로봇 다리 관절 구동용 서보모터
- **구매링크**: [AliExpress](https://ko.aliexpress.com/)

### DS51150-270 Servo Horn 5pcs

- **개수**: 4개
- **용도**: 서보모터와 로봇 관절 연결
- **구매링크**: [AliExpress](https://ko.aliexpress.com/i/1005007393245886.html)

### 6703ZZ Deep Groove Ball Bearing

- **개수**: 20개
- **용도**: 다리 관절 회전 보조
- **구매링크**: [Coupang](https://www.coupang.com/vp/products/6143283810?itemId=11802532451&vendorItemId=79076107061&q=6703ZZ&searchId=d0d0392a5285022&sourceType=search&itemsCount=36&searchRank=0&rank=0&traceId=mls5mr3f)

---

## 3. Mechanical Parts

### PLA Filament

- **개수**: 4개
- **용도**: 로봇 프레임 및 구조물 3D 프린팅
- **구매링크**: [Coupang](https://www.coupang.com/vp/products/6326227472?itemId=13209923756&vendorItemId=92097957520&q=pla+필라멘트&itemsCount=36&searchId=2c9f4d179589258&rank=3&searchRank=3)

### 스텐 육각렌치볼트 M4×10

- **개수**: 1봉
- **용도**: 로봇 조립용 볼트
- **구매링크**: [Coupang](https://www.coupang.com/vp/products/7877154060?utm_source=chatgpt.com)

---

## 4. Power System

### DXF Li-ion Battery 3S 7200mAh 100C

- **개수**: 2개
- **용도**: 서보모터 전원 공급
- **구매링크**: [RC9](https://rc9.co.kr/product/특가행사dxf-배터리-리튬-111v-7200mah-100c3s-dxf-한국총판-rc9-정품dxf01/80067/display/1/)

### DXF Li-ion Battery 2S 7200mAh 100C

- **개수**: 1개
- **용도**: 보드 및 센서 전원 공급
- **구매링크**: [AliExpress](https://ko.aliexpress.com/item/1005008268788282.html?gatewayAdapt=glo2kor)

### 고출력 스텝다운 5A 가변 DC 컨버터

- **개수**: 2개
- **용도**: 배터리 전압 강하용 DC-DC 컨버터
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=1321141)

### 니토 미니 배터리 잔량표시 볼트메타

- **개수**: 3개
- **용도**: 배터리 잔량 및 전압 확인
- **구매링크**: [Coupang](https://www.coupang.com/vp/products/334419473?itemId=1067530756&vendorItemId=75456141668&q=리튬배터리+3S+잔량표시&itemsCount=36&searchId=6dd464417351806&rank=2&searchRank=2)

### 아두이노 3채널 5V 릴레이 모듈

- **개수**: 2개
- **용도**: 각 다리 서보모터 전원 공급 및 차단
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=12147824)

### CANAL MR2-110-C5-BB Switch

- **개수**: 1개
- **용도**: 로봇 메인 전원 스위치
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=38183)

---

## 5. Connector / Wiring

### AMASS XT60E-F Female Connector

- **개수**: 3개
- **용도**: 배터리 고정 및 충전용 커넥터
- **구매링크**: [RCBank](https://www.rcbank.co.kr/m2/goods/view.php?goodsno=95419)

### JST-XH Balance Charge Wire

- **개수**: 3개
- **용도**: 배터리 밸런스잭 연장선
- **구매링크**: [HobbyZone](https://hobbyzone.kr/product/jst-xh-2s3s4s6s-20cm-22awg-balance-charge-wire-배터리-밸런스잭-연장선/5304/)
- **비고**: 2S용 1개, 3S용 2개

### XT60-F 암/수 커넥터

- **개수**: 3개
- **용도**: 배터리-전선 연결 소켓
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=1078005)

### 10AWG 실리콘 케이블

- **개수**: 10개
- **용도**: 메인 전원용 굵은 전선
- **구매링크**: [Coupang](https://www.coupang.com/vp/products/8537462031?itemId=24715590149&searchId=8327ed906c3b463ea0b5a524e92736ec&sourceType=brandstore_sdp_atf-best_products&storeId=10240&subSourceType=brandstore_sdp_atf-best_products&vendorId=A00502821&vendorItemId=90750092729)

### 18AWG 실리콘 케이블

- **개수**: 10개
- **용도**: 모터 전원용 전선
- **구매링크**: [Coupang](https://www.coupang.com/vp/products/8264903762?itemId=23813552001&searchId=6246d3a0cbd5455f8f25c7fffbb5633c&sourceType=brandstore_sdp_atf-best_products&storeId=10240&subSourceType=brandstore_sdp_atf-best_products&vendorId=A00502821&vendorItemId=90837339910)

### 22AWG × 4C 전선

- **개수**: 20개
- **용도**: 센서 및 신호용 얇은 전선
- **구매링크**: [Coupang](https://www.coupang.com/vp/products/7156740267?itemId=18003337265&searchId=492e19c0b188424685e18897d2f7e036&sourceType=brandstore_sdp_atf-best_products&storeId=34612&subSourceType=brandstore_sdp_atf-best_products&vendorId=A00340725&vendorItemId=85159319229#sdpReview)

### LiDAR Cable

- **개수**: 1개
- **용도**: Livox Mid-360 LiDAR와 Jetson Orin Nano Super 간 통신
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=12147824)

---

## 6. Sensor / Perception

### Livox Mid-360 LiDAR

- **개수**: 1개
- **용도**: 주변 지형 및 장애물 인식
- **구매링크**: [Coupang](https://www.coupang.com/vp/products/9250130341?itemId=27358998182&vendorItemId=94325064888&src=1042503&spec=10304025&addtag=400&ctag=9250130341&lptag=9250130341-27358998182&itime=20260219004855&pageType=PRODUCT&pageValue=9250130341&wPcid=17511806024806833762516&wRef=www.google.com&wTime=20260219004855&redirect=landing&gclid=&mcid=9ea861c46b59402eb063721237189ad6&campaignid=&adgroupid=)

### Intel RealSense D435

- **개수**: 1개
- **용도**: 주변 지형 및 깊이 정보 인식
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=13017227)

### WT931 9DOF IMU

- **개수**: 1개
- **용도**: 로봇의 기울기와 방향 측정
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=10886274)

### Interlink FSR 402 Pressure Sensor

- **개수**: 6개
- **용도**: 로봇 발의 지면 접촉 여부 확인
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=33870)

### ADC Module

- **개수**: 3개
- **용도**: STM32의 부족한 ADC 채널 보완
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=12147824)

### NEO-M8N GPS Module

- **개수**: 1개
- **용도**: 로봇의 실시간 위치 좌표 확인
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=15106909)

---

## 7. Communication / Remote Control

### RadioMaster Pocket ELRS 조종기

- **개수**: 1개
- **용도**: 로봇 수동 조종용 컨트롤러
- **구매링크**: [SUSUNG RC](https://susungrc.com/product/%EB%9D%BC%EB%94%94%EC%98%A4%EB%A7%88%EC%8A%A4%ED%84%B0-pocket-%EC%A1%B0%EC%A2%85%EA%B8%B0-elrs/2385/)

### ELRS Receiver Module

- **개수**: 1개
- **용도**: 조종기 신호를 수신하여 로봇 제어부로 전달
- **구매링크**: [DeviceMart](https://www.devicemart.co.kr/goods/view?no=15138136)

### RYLR998 UART LoRa Module

- **개수**: 2개
- **용도**: 로봇과 컴퓨터 간 장거리 무선 통신
- **구매링크**: [Coupang](https://www.coupang.com/vp/products/7232535124?itemId=18355967552&vendorItemId=85499819592&src=1042503&spec=10304025&addtag=400&ctag=7232535124&lptag=7232535124-18355967552&itime=20260228152610&pageType=PRODUCT&pageValue=7232535124&wPcid=17511806024806833762516&wRef=www.google.com&wTime=20260228152610&redirect=landing&gclid=CjwKCAiAnoXNBhAZEiwAnItcG2BsnsktZJzYX85kpdk3TqxUEqCcQUz-7MKTnwaD-J5tWiEd4ZkqYhoC9DkQAvD_BwE&mcid=c32dedb64c9944bcab7d012420c4f76a&campaignid=22815108882&adgroupid=)
