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
