<template>
  <div class="prose-table-scroll" tabindex="0" role="region" aria-label="文章表格，可横向滚动">
    <table v-bind="$attrs"><slot /></table>
  </div>
</template>

<script setup lang="ts">
defineOptions({ inheritAttrs: false })
</script>
