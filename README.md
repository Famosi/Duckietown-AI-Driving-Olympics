


## Table of contents
* [Introduction](#introduction)
* [Setup]()
* [Status]()


# Duckietown - AI Driving Olympics
<a href="http://aido.duckietown.org"><img width="200" src="https://www.duckietown.org/wp-content/uploads/2018/12/AIDO_no_text-e1544555660271.png"/></a>

The [“AI Driving Olympics” (AI-DO)](http://aido.duckietown.org/) is a competition with the objective of evaluating the state of the art in machine learning and artificial intelligence for mobile robotics.

The AI Driving Olympics competition is structured into the following separate challenges:
* Lane Following - `LF`
* Lane Following + Vehicles - `LFV`
* Lane following with vehicles and intersections- `LFVI`
* Autonomous Mobility-on-Demand - `AMoD`

This project is a solution for the `LF` challenge: control of a Duckiebot to drive on the right lane on streets within Duckietown without other moving Duckiebots present. 

##Overview
The approach is to build an "expert" using reinforcement learning (RL) and use this expert to collect data.
Collected data are used to train a neural network and build a model to predict, at each step, which action to take.

The expert, at each step, produces a prediction tree and takes the action that maximize the reward.

```
# Compute reward
reward = (
        + speed * self.cof_speed
        + align * self.cof_align
        - dist * self.cof_dist
        + not_derivable
)
```

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
See deployment for notes on how to deploy the project on a live system.

### Installing

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc


