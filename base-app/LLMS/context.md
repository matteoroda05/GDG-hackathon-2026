# LLM Project Context: Rubbish Detection System for Eco-Island Hackathon

## Project Overview

We want to create a rubbish detection system based on the **Luxonis OAK 4 D Pro** camera for a hackathon.

The camera features and usage instructions are provided in:

* `LLMS/luxonis-llms.md`

The project idea is designed for use in an **eco-island** system. A reference example of this type of organisation can be found here:

* https://ecoctrl.com/en/

## Eco-Island Use Case

The system is intended to work in an eco-island where:

1. The user identifies themselves through an authentication or identification system.
2. The user places the waste item in a dedicated detection area.
3. The detection system analyses the waste.
4. The system determines which bin the waste should be thrown into.
5. The system sends the detected waste category to the rest of the eco-island infrastructure.

For this project, assume that the physical system beyond the detection step is already working. The main responsibility of this project is to identify the correct waste category and inform the rest of the system.

## Primary Feature: Waste Detection and Classification

The first feature is the detection of the inserted object and classification into the correct waste category.

The target categories are:

* Plastic
* Metal
* Paper
* Glass
* Organic
* Generic

The system should detect the object placed in the detection area and decide which bin category it belongs to.

## Secondary Feature: Wrong-Bin Detection

The secondary idea is to check whether the rubbish was thrown into the correct bin.

If the item was thrown into the wrong bin, the system should:

1. Signal an error.
2. Track the position of the incorrectly placed item.
3. Make the item easier to locate later, even if it becomes covered by other rubbish.

## Depth Perception and Bin-Fill Estimation

We also want to add depth perception or volume estimation for each item thrown away.

This should be used to estimate whether a bin is full.

If a bin is full, the system should notify the responsible company or operator.

## Web App Requirement

For the hackathon, we also need to create a web app to showcase the product.

The web app should demonstrate the system’s core functionality and make the project understandable to judges, users, and stakeholders.

## Presentation Requirement

For the hackathon, we also need to prepare a presentation explaining the product.

The presentation should communicate:

* The problem being solved
* The proposed solution
* The use of the Luxonis OAK 4 D Pro
* The waste classification workflow
* The eco-island use case
* The technical architecture
* The business and sustainability value
* The current prototype scope
* Possible future developments

## Optional Feature: Brand Detection

An additional possible feature is the detection of different brands for specific products.

Example:

* Detecting the brand of aluminium cans

This could be used to collect data for:

* Advertising insights
* Business applications
* Recycling analytics
* Product consumption statistics

## Development Requirement

The development process must be:

* Modular
* Testable throughout the project
* Suitable for iterative development during a hackathon

Each component should be designed so that it can be tested independently.

Possible modules may include:

* Camera input module
* Object detection module
* Waste classification module
* Depth and volume estimation module
* Bin-fill estimation module
* Wrong-bin detection module
* Brand recognition module
* Backend API module
* Web app module
* Dashboard or notification module
* Presentation/demo module

## Important Instruction for the LLM

Do not resolve ambiguities independently.

Whenever a requirement is unclear, incomplete, or has multiple possible interpretations, ask the user for confirmation before making a decision.

Do not assume implementation details unless they have been explicitly confirmed by the user.

## Goal

The goal is to develop a modular, testable rubbish detection system using the Luxonis OAK 4 D Pro camera for an eco-island hackathon project.

The system should classify waste into the correct disposal category, support wrong-bin detection, use depth or volume estimation to monitor bin fullness, and include a web app and presentation for the hackathon showcase.
