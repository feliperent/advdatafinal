{# Override dbt's default schema-naming behavior so that:
   - if a model has a +schema config, that value is used LITERALLY (no profile-schema prefix)
   - if no schema config, fall back to the default schema from profiles.yml
   This matches the midterm convention where schemas are: raw, datos_masked, silver, gold. #}

{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
