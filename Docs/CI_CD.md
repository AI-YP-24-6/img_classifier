Для упрощения и автоматизации процессов разработки настроили CI/CD.

## GitHub Actions pre-commit

Автоматическая проверка кода используя уже настроенный `.pre-commit-config.yaml`:

``` yaml linenums="1"
--8<-- "./.github/workflows/pre-commit.yml"
```
Если были обнаружены ошибки, то скрип их исправит и сделает коммит в ветку с изменениями.

## GitHub Pages

Автоматический деплой сайта на прямо в github pages после коммита в ветку main:

``` yaml linenums="1"
--8<-- "./.github/workflows/gh-pages.yml"
```
Статичный сайт доступен по адресу [https://AI-YP-24-6.github.io/img_classifier/](https://AI-YP-24-6.github.io/img_classifier/).

## Docker

Для упрощения проверки докер файлов используется [hadolint](https://github.com/hadolint/hadolint)

``` yaml linenums="1"
--8<-- "./.github/workflows/docker-linter.yml"
```
Если ошибок нет, то скрип напишет сообщение "No docker📦 errors found 🎉✨" в комментарии к pull request.

## Pycodestyle & Pylint

Автоматическая проверка кода на соответствие стандартам [pycodestyle](https://pep8.readthedocs.io/en/latest/) и [pylint](https://pylint.pycqa.org/en/latest/)

``` yaml linenums="1"
--8<-- "./.github/workflows/linters.yml"
```
Если были обнаружены ошибки, то скрип их исправит и сделает комментарий в pull request.

## Notebook-formater

У ноутбуков из Google colab есть раздел metadata, которые не может обработать nbconvert. Jq скрипт очищает файл от metadata.<br>
Так же скрипт делает корректные execution_count ячеек

``` yaml linenums="1"
--8<-- "./.github/workflows/notebook-formater.yml"
```

Если были обнаружены ошибки, то скрип их исправит и сделает коммит в pull request.
