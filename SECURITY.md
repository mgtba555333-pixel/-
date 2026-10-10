# Security and Responsible Use

PulmoSight AI is a research prototype, not a medical device. Do not expose it to the public internet with real patient images.

Before deployment:
- Use HTTPS and appropriate authentication/access controls.
- Set upload limits and rate limits at both the application and hosting layers.
- Keep model weights and secrets out of Git.
- Review the hosting provider's request buffering, logs, and retention behavior.
- Avoid identifiable patient data in issues, pull requests, fixtures, screenshots, logs, or bug reports.
- Confirm data and model license terms before redistribution.
- Obtain independent security, privacy, and clinical review before any real-world patient use.

To report a security issue, avoid posting sensitive details publicly. Contact the repository maintainer through GitHub's private security reporting feature if available.
