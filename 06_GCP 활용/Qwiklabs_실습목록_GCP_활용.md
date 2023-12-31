### Google Qwiklabs (Cloud Skills Boost) 실습 목록 : GCP 활용
---
[Course 01] : Google Cloud Big Data and Machine Learning Fundamentals

https://www.cloudskillsboost.google/course_templates/3<br>
에 접속해서 과정 목록 나오기 전의 상단의 "Enroll in this on-demand course"을 눌러 등록한 다음 아래 실습을 수행한다

<1st Day><br>
[1] Big Data and Machine Learning on Google Cloud : (Course 01)<br>
   [LAB] : Exploring a BigQuery Public Dataset : (BigQuery 기초)

---

> [Main Course] : Data Engineering on Google Cloud
      https://www.cloudskillsboost.google/journeys/16

[Course 02] : Modernizing Data Lakes and Data Warehouses with Google Cloud<br>
https://www.cloudskillsboost.google/course_templates/54<br>
에 접속해서 과정 목록이 보이기 전의 상단에나타나기 보이기 전에 "Enroll in this on-demand course"을 눌러 <br>
코스를 수강 등록한 다음 아래 실습을 수행한다


[2] Introduction to Data Engineering : (Course 02)<br>
   [LAB] :  Using BigQuery to do Analysis : (BigQuery 활용)


<2nd Day><br>
[3] Building a Data Lake : (Course 02)<br>
   [LAB] : Loading Taxi Data into Google Cloud SQL 2.5   : (Cloud SQL 기초)


[4] Building a Data Warehouse : (Course 02)<br>
   [LAB] : Loading data into BigQuery     : (BigQuery 활용)


[5] Building a Data Warehouse : (Course 02)<br>
   [LAB] : Working with JSON and Array data in BigQuery 2.5 : (BigQuery 고급)


---

[Course 03] : Building Batch Data Pipelines on Google Cloud<br>
https://www.cloudskillsboost.google/course_templates/53<br>
에 접속해서 과정 목록이 보이기 전의 상단에나타나기 보이기 전에 "Enroll in this on-demand course"을 눌러 <br>
코스를 수강 등록한 다음 아래 실습을 수행한다

<3rd Day><br>
[6] Executing Spark on Dataproc : (Course 03)<br>
   [LAB] : Running Apache Spark jobs on Cloud Dataproc  : (Dataproc 기본)



<4th Day><br>
[7] Serverless Data Processing with_Dataflow : (Course 03)<br>
   [Lab] : A Simple Dataflow Pipeline (Python)    :  (Dataflow 기본)

- 퀵랩환경이 아닌경우에는 "training-vm" 가상머신을  생성한 다음 SSH접속 후 아래 설치 명령을 먼저 수행한다<br>
[주의]:VM생성시 [ID 및 API 액세스]의 [액세스 범위]에서 [모든 Cloud API에 대한 전체 액세스 허용]을 선택해주고 생성한다

[Debian git 설치]<br>
sudo apt update<br>
sudo apt install git<br>

[가상머신에 python3과 apache-beam설치]<br>
sudo apt-get -y install python3-pip<br>
pip install apache-beam[gcp]<br>

BUCKET="jnu-idv-xx"<br>
echo $BUCKET<br>

<5th Day><br>
[8] Serverless Data Processing with_Dataflow : (Course 03)<br>
   [Lab] : MapReduce in Beam (Python)]   :  (Dataflow 기본)<br>
  

[9] Serverless Data Processing with_Dataflow : (Course 03)<br>
   [Lab] : Side Inputs (Python)]  :  (Dataflow 기본)<br>

- 퀵랩환경이 아닌경우에는 "training-vm" 가상머신을  생성한 다음 SSH접속 후 아래 설치 명령을 먼저 수행한다<br>
[주의]:VM생성시 [ID 및 API 액세스]의 [액세스 범위]에서 [모든 Cloud API에 대한 전체 액세스 허용]을 선택해주고 생성한다<br>

[Debian git 설치]<br>
sudo apt update<br>
sudo apt install git<br>

[가상머신에 python3과 apache-beam설치]<br>
sudo apt-get -y install python3-pip<br>
pip install apache-beam[gcp]<br>

BUCKET="jnu-idv-xx"<br>
echo $BUCKET<br>

---

[Course 04] : Building Resilient Streaming Analytics Systems on GCP<br>
https://www.cloudskillsboost.google/course_templates/52<br>
에 접속해서 과정 목록이 보이기 전의 상단에나타나기 보이기 전에 "Enroll in this on-demand course"을 눌러 <br>
코스를 수강 등록한 다음 아래 실습을 수행한다<br>

[10] [Lab] : Google Cloud Pub/Sub: Qwik Start - Console  : (Pub/Sub 기초)<br>
https://www.cloudskillsboost.google/focuses/3719?catalog_rank=%7B%22rank%22%3A1%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=25243204

[11] [Lab] : Google Cloud Pub/Sub: Qwik Start - Command Line : (Pub/Sub 기초)<br>
https://www.cloudskillsboost.google/focuses/925?catalog_rank=%7B%22rank%22%3A2%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=25243204

export DEVSHELL_PROJECT_ID=$(gcloud config get-value project)<br>


<6th Day><br>
[12] Serverless Messaging with Pub Sub  : (Course 04)  <br>
   [Lab] : Streaming Data Processing: Publish Streaming Data into PubSub    : (Pub/Sub 활용)<br>
(파이썬 버전 문제로 Qwiklab에서 실습 진행 할것)

 [PubSub v2.0 변경사항(python 3.6이상)]<br>
 https://cloud.google.com/python/docs/reference/pubsub/latest/upgrading

 [최신 Pubsub python 소스 예제]<br>
 https://github.com/googleapis/python-pubsub/tree/main/samples/snippets<br>
 [Lab] Google Cloud Pub/Sub: Qwik Start - Python<br>
 https://www.cloudskillsboost.google/focuses/2775?catalog_rank=%7B%22rank%22%3A1%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=25321450

[13] Dataflow Streaming Features : (Course 04)  <br>
   [Lab] : Streaming Data Processing: Streaming Data Pipelines   : (Dataflow / Pub/Sub / BigQuery 연동)<br>
(파이썬 버전 문제로 Qwiklab에서 실습 진행 할것)


[14] High-Throughput BigQuery and Bigtable Streaming Features : (Course 04) <br>
   [Lab] : Streaming Data Processing: Streaming Analytics and Dashboards : (BigQuery / Looker Studio 시각화) 


[15] High-Throughput BigQuery and Bigtable Streaming Features : (Course 04)<br>
   [Lab] : Streaming Data Pipelines into Bigtable   : (Dataflow / Pub/Sub / Bigtable 연동)<br>
(파이썬 버전 문제로 Qwiklab에서 실습 진행 할것)

* HBase Shell 명령어 설명<br>
1. create '테이블이름','column 이름' : 새로운 테이블을 생성<br>
2. scan '테이블이름' : 테이블의 데이터를 scan한다(쿼리하여 출력)<br>
3. list : 테이블 목록을 보여줌<br>
4. get '테이블이름','row이름' : table에서 1개 row만 출력<br>

* HBase Shell 명령어 실습<br>
create 'my-table', 'cf1'<br>
list<br>
put 'my-table','r1','cf1:c1','test-value'<br>
scan 'my-table'<br>
get 'my-table','r1'<br>
disable 'my-table'<br>
drop 'my-table'<br>
exit<br>




[16] Advanced BigQuery Functionality and Performance : (Course 04)  --> 실습 생략<br>
   [Lab]: Optimizing your BigQuery Queries for Performance : (BigQuery) 

[17] Advanced BigQuery Functionality and Performance : (Course 04)   --> 실습 생략<br>
   [Lab]: Partitioned Tables in BigQuery : (BigQuery) 

