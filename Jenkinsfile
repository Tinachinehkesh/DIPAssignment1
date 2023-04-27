pipeline {
    agent { docker { image 'pmantini/assignment-cosc6380:latest' } }

    environment {
        PATH = "env/bin/:$PATH"
    }
    stages {
        stage('build') {
            steps {
                sh 'python dip_hw1_distortion.py -i kenny.jpg'
                sh 'python dip_hw1_distortion.py -i kenny.jpg -m bilinear'
                sh 'python dip_hw1_distortion.py -i kenny.jpg -k 0.0005'
                sh 'python dip_hw1_distortion.py -i kenny.jpg -k 0.0005 -m bilinear'
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'output/**/*.* ', onlyIfSuccessful: true
        }
    }
}