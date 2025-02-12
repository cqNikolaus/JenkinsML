pipeline {
  agent any
  stages {
    stage('Checkout') {
      steps {
        git 'https://github.com/cqNikolaus/JenkinsML.git'
      }
    }
    stage('Build Docker Image') {
      steps {
        sh 'docker build -t jenkinsml-poc .'
      }
    }
    stage('Run Container') {
      steps {
        withCredentials([string(credentialsId: 'jenkins-api-token', variable: 'JENKINS_TOKEN')]) {
          sh 'docker run --rm -e JENKINS_TOKEN=${JENKINS_TOKEN} jenkinsml-poc'
        }
      }
    }
  }
}
