# neurorse workshops

This repo hosts static site(s). The source materials are hosted elsewhere, the output is built by Jenkins and pushed here, and the site itself is hosted on github pages. The software architecture comes from [jenkins-webserver-template](https://github.com/flatironinstitute/jenkins-webserver-template).

> [!CAUTION]
> This repo should not be edited manually.
> 
> Every week, it will delete the documentation corresponding to pull requests that have been deleted and over-write the git history. Thus, the history will diverge with any local copies --- you should delete your local copy and re-clone this as needed.

We use this repo essentially as a webserver: the html is placed here and then
hosted on github pages, and we treat this repo as "shallow", not tracking the
history. The building and pushing is all handled by Jenkins jobs triggered in
the main repo.

