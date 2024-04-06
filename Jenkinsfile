#!groovy
def BN = (BRANCH_NAME == 'master' || BRANCH_NAME.startsWith('releases/')) ? BRANCH_NAME : 'releases/2024-06'

// The knime version defines which parent pom is used in the built feature. The version was added in KNIME 4.7.0 and
// will have to be updated with later KNIME versions.
def knimeVersion = '5.3'

def repositoryName = "knime-python-sklearn"

def extensionPath = "." // Path to knime.yml file
def outputPath = "output"

library "knime-pipeline@$BN"

properties([
    parameters(
        [p2Tools.getP2pruningParameter()] + \
        workflowTests.getConfigurationsAsParameters() + \
        condaHelpers.getForceCondaBuildParameter()
    ),
    buildDiscarder(logRotator(numToKeepStr: '5')),
    disableConcurrentBuilds()
])

try {
    timeout(time: 100, unit: 'MINUTES') {
        node('workflow-tests && ubuntu22.04 && java17') {
            stage("Checkout Sources") {
                env.lastStage = env.STAGE_NAME
                checkout scm
            }

            stage("Create Conda Env"){
                env.lastStage = env.STAGE_NAME
                prefixPath = "${WORKSPACE}/${repositoryName}"
                condaHelpers.createCondaEnv(prefixPath: prefixPath, pythonVersion:'3.9', packageNames: ['knime-extension-bundling'])
            }
            stage("Build Python Extension") {
                env.lastStage = env.STAGE_NAME
                force_conda_build = params?.FORCE_CONDA_BUILD ? "--force-new-timestamp" : ""

                withMavenJarsignerCredentials(options: [artifactsPublisher(disabled: true)], skipJarsigner: false) {
                    withEnv([ "MVN_OPTIONS=-Dknime.p2.repo=https://jenkins.devops.knime.com/p2/knime/" ]) {
                        withCredentials([usernamePassword(credentialsId: 'ARTIFACTORY_CREDENTIALS', passwordVariable: 'ARTIFACTORY_PASSWORD', usernameVariable: 'ARTIFACTORY_LOGIN'),
                        ]) {
                            sh """
                            micromamba run -p ${prefixPath} build_python_extension.py ${extensionPath} ${outputPath} -f --knime-version ${knimeVersion} --knime-build --excluded-files ${prefixPath} ${force_conda_build}
                            """
                        }
                    }
                }
            }
            stage("Deploy p2") {
                    env.lastStage = env.STAGE_NAME
                    p2Tools.deploy(outputPath)
                    println("Deployed")

                }
            workflowTests.runTests(
                dependencies: [
                    repositories: [
                        'knime-python',
                        'knime-python-types',
                        'knime-core-columnar',
                        'knime-testing-internal',
                        'knime-python-legacy',
                        'knime-conda',
                        'knime-python-bundling',
                        'knime-credentials-base',
                        'knime-gateway',
                        'knime-base',
                        repositoryName
                        ],
                ],
            )
        }
    }
} catch (ex) {
    currentBuild.result = 'FAILURE'
    throw ex
} finally {
    notifications.notifyBuild(currentBuild.result)
}
