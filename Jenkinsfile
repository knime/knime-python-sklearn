#!groovy
def BN = (BRANCH_NAME == 'master' || BRANCH_NAME.startsWith('releases/')) ? BRANCH_NAME : 'releases/2025-07'

def repositoryName = 'knime-python-sklearn'

library "knime-pipeline@$BN"

properties([
    /*
    When changes occur in the upstream jobs (e.g., "knime-python"), this configuration 
    ensures that dependent jobs (e.g., "knime-python-sklearn") are automatically rebuilt.

    Example:
        upstream(
            'knime-abc/' + env.BRANCH_NAME.replaceAll('/', '%2F') +
            ', knime-xyz/' + env.BRANCH_NAME.replaceAll('/', '%2F')
        )
    */
    pipelineTriggers([
		upstream('knime-python/' + env.BRANCH_NAME.replaceAll('/', '%2F'))
	]),
    parameters(
        knimetools.getPythonExtensionParameters()
    ),
    buildDiscarder(logRotator(numToKeepStr: '5')),
    disableConcurrentBuilds()
])

try {
    knimetools.defaultPythonExtensionBuild()

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
