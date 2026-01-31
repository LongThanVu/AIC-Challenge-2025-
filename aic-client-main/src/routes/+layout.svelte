<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>AIC</title>
</svelte:head>

<div class="container flex min-h-screen min-w-screen flex-col items-center gap-2 p-2">
	{@render children?.()}
</div>
