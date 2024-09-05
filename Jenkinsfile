#!groovy
def BN = (BRANCH_NAME == 'master' || BRANCH_NAME.startsWith('releases/')) ? BRANCH_NAME : 'releases/2024-12'

def repositoryName = 'knime-python-sklearn'

library "knime-pipeline@$BRANCH_NAME"

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
    knimetools.defaultPythonExtensionBuild(pythonVersion: '3.9',)

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
                'knime-productivity-oss',
                'knime-json',
                'knime-javasnippet',
                'knime-reporting',
                'knime-filehandling',
                'knime-scripting-editor',
                repositoryName
                ],
        ],
    )
} catch (ex) {
    currentBuild.result = 'FAILURE'
    throw ex
} finally {
    notifications.notifyBuild(currentBuild.result)
}
