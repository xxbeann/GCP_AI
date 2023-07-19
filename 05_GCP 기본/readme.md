### Google Qwiklabs (Cloud Skills Boost) 실습 목록
---

https://www.cloudskillsboost.google/
(처음 30일간 무료 , 월간 구독료 $29)

<1st Day>
[1] A Tour of Google Cloud Hands-on Labs : (Qwiklab 사용법 설명)

https://www.cloudskillsboost.google/focuses/2794?locale=ko&parent=catalog


[2] Creating a Virtual Machine  : (Compute Engine 기초)

https://www.cloudskillsboost.google/focuses/3563?catalog_rank=%7B%22rank%22%3A1%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=24944033

gcloud config set compute/region "region=us-central1"
gcloud config get compute/region

export REGION=region=us-central1
echo $REGION

export ZONE=us-central1-a
echo $ZONE


[3] Getting Started with Cloud Shell and gcloud  : (Cloud Shell 과 gcloud 사용법)

https://www.cloudskillsboost.google/focuses/563?parent=catalog



<2nd Day>
[4] Cloud Storage: Qwik Start - Cloud Console   :  (Cloud Storage 기초)

https://www.cloudskillsboost.google/focuses/1760?catalog_rank=%7B%22rank%22%3A1%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalonvm install --ltsg&search_id=24944290


[5] Cloud Storage: Qwik Start - CLI/SDK : (Cloud Storage 기초)

https://www.cloudskillsboost.google/focuses/569?catalog_rank=%7B%22rank%22%3A2%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=24970560

 acl ch --> acl change (https://download.huihoo.com/google/gdgdevkit/DVD1/developers.google.com/storage/docs/gsutil/commands/acl.html)


[6] Hosting a Web App on Google Cloud Using Compute Engine : (Compute Engine/Cloud Storage/Load Balancers/CDN)
   여기서는  "Task 4. Create Compute Engine instances" 까지만 실습 수행한다 (Instance Group과 LB/CDN은 숙제)

https://www.cloudskillsboost.google/focuses/11952?catalog_rank=%7B%22rank%22%3A1%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=24944216

(힌트) gcloud 명령 사용 시 "REGION" 과 Region부분에  모두 us-central1 으로  Zone은 us-central1-f 로 입력한다

LF/CRLF  --> https://velog.io/@dev_yong/CRLF%EC%99%80-LF%EC%B0%A8%EC%9D%B4%EC%9D%98-%EC%9D%B4%ED%95%B4
project = augmented-path-392404
watch -n 2 curl http://34.123.142.162:8080

 *** Windows에 Cloud SDK를 설치하고 gcloud로 vm 인스턴스 생성해본 후 다음 실습으로 진행 


<3rd Day>
[7]  Cloud IAM: Qwik Start  : (IAM 기초)

https://www.cloudskillsboost.google/focuses/44159?catalog_rank=%7B%22rank%22%3A1%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=24971369

[8] Service Accounts and Roles : Fundamentals : (서비스 계정과 Role)

https://www.cloudskillsboost.google/focuses/1038?catalog_rank=%7B%22rank%22%3A2%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&locale=ko&parent=catalog&search_id=24999935


 *** VPC Networking 이론 설명후 실습 진행

[9] Set Up Network and HTTP Load Balancers : (HTTP Load Balancer)

https://www.cloudskillsboost.google/focuses/12007?catalog_rank=%7B%22rank%22%3A2%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=24971158

[주의] Task 5에서 3.Create the fw-allow-health-check firewall rule에서 아래처럼 --source-ranges옵션을 생략해준다
gcloud compute firewall-rules create fw-allow-health-check \
  --network=default \
  --action=allow \
  --direction=ingress \  
  --target-tags=allow-health-check \
  --rules=tcp:80

[아래 명령으로 얻어진 주소를 웹브라우저에 입력하고 다시 고침을 하면 출력 값이 변한다] 
gcloud compute addresses describe lb-ipv4-1 \
  --format="get(address)" \
  --global

[마지막 backend 접속 출력] 
Page served from: lb-backend-group-z7h6 에서  Page served from: lb-backend-group-92fj 으로 변한다

[Compute Engine ->인스턴스 그룹-> lb-backend-group:EDIT에서 인스턴스 수를 3으로 수정하면 3개의 VM인스턴스가 생성된다]


[10] Networking 101 : (기본 네트워크 설정)
  (Task 6까지만 수행한다)
https://www.cloudskillsboost.google/focuses/1743?catalog_rank=%7B%22rank%22%3A10%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=24999434
 
[Set your region and zone 마지막 줄에 zone이 아니라  region임]
gcloud config set compute/zone "us-central1-a"
export ZONE=$(gcloud config get compute/zone)
gcloud config set compute/region "us-central1"
export REGION=$(gcloud config get compute/region)   


[11]  VPC Networking Fundamentals : (VPC 기초)

https://www.cloudskillsboost.google/focuses/1229?catalog_rank=%7B%22rank%22%3A1%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=24974993


[12] Multiple VPC NetworksMultiple VPC Networks : (다중 VPC 네트워크)

https://www.cloudskillsboost.google/focuses/1230?catalog_rank=%7B%22rank%22%3A2%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&locale=ko&parent=catalog&search_id=24977321


[13] VPC Network Peering : (VPC Peering)

https://www.cloudskillsboost.google/focuses/964?catalog_rank=%7B%22rank%22%3A1%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=24999254

[14] VPC Networks - Controlling Access : (방화벽 규칙)

https://www.cloudskillsboost.google/focuses/1231?catalog_rank=%7B%22rank%22%3A10%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&locale=ko&parent=catalog&search_id=24999491
