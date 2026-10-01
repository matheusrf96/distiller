name: Pull Request
description: Template for pull requests to standardize descriptions and checklists

body:
  - type: markdown
    attributes:
      value: |
        Thanks for contributing! Please fill out the information below to help us understand and review your changes.

  - type: input
    id: spec-id
    attributes:
      label: Spec ID
      description: If this PR implements a spec, reference it here (e.g., SDD-0003). For trivial fixes, enter "N/A".
      placeholder: "SDD-XXXX or N/A"
    validations:
      required: true

  - type: checkboxes
    id: ac-coverage
    attributes:
      label: Acceptance Criteria Covered
      description: Check the ACs this PR satisfies. Required for spec'd PRs. For N/A specs, check the first box only.
      options:
        - label: N/A — this PR is a trivial fix with no spec
          required: false
        - label: AC1
          required: false
        - label: AC2
          required: false
        - label: AC3
          required: false
        - label: AC4
          required: false
        - label: AC5
          required: false
        - label: AC6
          required: false
        - label: AC7
          required: false
        - label: AC8
          required: false
        - label: AC9+
          required: false

  - type: textarea
    id: description
    attributes:
      label: Description
      description: Describe the changes you've made. What problem does this PR solve?
      placeholder: |
        - What are the key changes?
        - Why were these changes necessary?
        - Which spec (SDD-XXXX) does this implement?
    validations:
      required: true

  - type: textarea
    id: changes
    attributes:
      label: Changes Made
      description: List the specific files and changes modified
      placeholder: |
        - src/distiller/synthesis/qa.py: Added grounded QA generation
        - tests/unit/test_synthesis_qa.py: Added parsing and prompt tests
    validations:
      required: true

  - type: textarea
    id: testing
    attributes:
      label: Testing
      description: How have you tested these changes?
      placeholder: |
        - [ ] Unit tests added/updated
        - [ ] Integration tests added/updated
        - [ ] Manual smoke test on a real fixture
    validations:
      required: true

  - type: dropdown
    id: type
    attributes:
      label: Type of Change
      description: What type of change is this?
      options:
        - Bug fix
        - Feature
        - Enhancement
        - Documentation
        - Performance improvement
        - Refactoring
        - Dependency update
    validations:
      required: true

  - type: dropdown
    id: breaking
    attributes:
      label: Breaking Changes
      description: Does this PR introduce any breaking changes?
      options:
        - "No breaking changes"
        - "Yes - breaking changes (see description)"
    validations:
      required: true

  - type: checkboxes
    id: checklist
    attributes:
      label: Pre-submission Checklist
      description: Please ensure all items are completed before submitting
      options:
        - label: My code follows the project's code style
          required: true
        - label: I have updated documentation if needed
          required: false
        - label: Tests pass locally (`make test`)
          required: true
        - label: Linting and typing pass (`make format`)
          required: true
        - label: Spec tree is valid (`make spec-check`)
          required: true
        - label: Coverage threshold is met (95%+)
          required: true
        - label: I have self-reviewed my own code
          required: true
        - label: No new warnings have been generated
          required: true
