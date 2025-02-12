pipeline {
  agent any
  stages {
    stage('Prepare Workspace') {
      steps {
        sh 'mkdir -p output'
      }
    }
    stage('Build Docker Image') {
      steps {
        sh 'docker build -t jenkins-builddata .'
      }
    }
    stage('Run Container') {
      steps {
        withCredentials([string(credentialsId: 'jenkins-api-token', variable: 'JENKINS_TOKEN')]) {
          sh 'docker run --rm -e JENKINS_TOKEN=${JENKINS_TOKEN} -v $WORKSPACE/output:/output jenkins-builddata'
        }
      }
    }
  }
  post {
    always {
      archiveArtifacts artifacts: 'output/build_data.csv', fingerprint: true
    }
  }
}
