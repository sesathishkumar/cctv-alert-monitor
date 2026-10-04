const $ = selector => document.querySelector(selector);
let seen = null;
let audio = null;
let feedConnected = false;
$('#mode').addEventListener('change', event => {
  $('#confidence-control').hidden = event.target.value === 'motion';
  $('[name="confidence"]').value = ['violence', 'combined'].includes(event.target.value) ? '0.7' : '0.6';
  if (['violence', 'combined'].includes(event.target.value)) {
    $('#input-source').value = 'video';
    $('#input-source').dispatchEvent(new Event('change'));
  }
});
api('/api/capabilities').then(data => {
  $('#weapons-option').disabled = !data.weapons.configured;
  $('#weapons-option').textContent = data.weapons.configured ? 'Possible guns + knives' : 'Guns + knives · model needed';
  $('#model-status').textContent = data.weapons.message;
  $('#violence-option').disabled = !data.violence.configured;
  $('#violence-option').textContent = data.violence.configured ? 'Possible fighting / physical violence' : 'Possible fighting · model needed';
  $('#violence-model-status').textContent = data.violence.message;
  $('#combined-option').disabled = !(data.weapons.configured && data.violence.configured);
}).catch(error => {$('#model-status').textContent = error.message;});
$('#input-source').addEventListener('change', event => {
  const upload = event.target.value === 'video';
  $('#upload-control').hidden = !upload;
  $('#video').required = upload;
});
$('#sound').addEventListener('change', async event => {
  if (event.target.checked) {
    audio ||= new (window.AudioContext || window.webkitAudioContext)();
    await audio.resume();
  }
});
function beep() {
  if (!audio || !$('#sound').checked) return;
  const oscillator = audio.createOscillator();
  const gain = audio.createGain();
  oscillator.connect(gain); gain.connect(audio.destination);
  oscillator.frequency.value = 740; gain.gain.value = 0.08;
  oscillator.start(); oscillator.stop(audio.currentTime + 0.2);
}
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
$('#settings').addEventListener('submit', async event => {
  event.preventDefault(); $('#start').disabled = true; $('#message').textContent = '';
  const data = new FormData(event.target);
  data.set('zone', JSON.stringify(['x','y','w','h'].map(key => Number(data.get(key))/100)));
  try {
    await api('/api/start', {method:'POST', body:data});
    $('#message').textContent = data.get('mode') !== 'motion'
      ? 'Model monitoring requested. Fighting mode needs about two seconds of video per window and repeated detections.'
      : 'Monitoring started. Background calibration takes 20 frames.';
  } catch (error) {$('#message').textContent = error.message;}
  finally {$('#start').disabled = false;}
});
$('#stop').addEventListener('click', async () => {
  try {await api('/api/stop', {method:'POST'}); $('#message').textContent = 'Monitoring stopped.';}
  catch (error) {$('#message').textContent = error.message;}
});
function text(tag, content, className) {
  const node = document.createElement(tag); node.textContent = content;
  if (className) node.className = className;
  return node;
}
async function update() {
  try {
    const [state, alerts] = await Promise.all([api('/api/status'), api('/api/alerts')]);
    $('#status').textContent = state.status.toUpperCase();
    const active = state.status === 'monitoring';
    $('#status').classList.toggle('active', active);
    $('#source-label').textContent = state.source || 'No active input';
    $('#frame-count').textContent = `${state.frames} frames processed`;
    $('#movement').textContent = state.movement ? 'Detection candidates present' : 'No current detection candidates';
    $('#feed-label').textContent = state.status.toUpperCase();
    $('#feed-label').hidden = !state.frames;
    if (state.frames && !feedConnected) {
      $('#feed').src = '/stream'; $('#feed').hidden = false; $('#empty').hidden = true;
      feedConnected = true;
    }
    if (state.error) $('#message').textContent = state.error;
    const current = new Set(alerts.map(alert => alert.id));
    if (seen && alerts.some(alert => !seen.has(alert.id))) beep();
    seen = current;
    $('#alert-count').textContent = alerts.length;
    const container = $('#alerts'); container.replaceChildren();
    if (!alerts.length) container.append(text('div', 'No alerts yet. Start monitoring to capture movement events.', 'empty-alerts'));
    for (const alert of alerts) {
      const row = text('article', '', 'alert');
      const link = document.createElement('a'); link.href = `/evidence/${alert.evidence}`;
      link.target = '_blank'; link.rel = 'noopener';
      const image = document.createElement('img'); image.src = link.href; image.alt = 'Captured movement evidence';
      link.append(image); row.append(link);
      const details = document.createElement('div');
      details.append(text('h3', alert.event), text('p', alert.source),
        text('span', new Date(alert.created).toLocaleString(), 'time'));
      if (alert.confidence != null) details.append(text('p', `Model confidence: ${Math.round(alert.confidence * 100)}% · human review required`));
      if (alert.clip) {
        const clip = text('a', `Download evidence clip · ${alert.clip_start.toFixed(1)}–${alert.clip_end.toFixed(1)}s`);
        clip.href = `/evidence/${alert.clip}`; clip.setAttribute('download', '');
        details.append(clip);
      }
      row.append(details);
      if (alert.acknowledged) row.append(text('span', '✓ Acknowledged', 'reviewed'));
      else {
        const button = text('button', 'Acknowledge');
        button.addEventListener('click', async () => {
          button.disabled = true;
          try {await api(`/api/alerts/${alert.id}/acknowledge`, {method:'POST'});}
          catch (error) {$('#message').textContent = error.message;}
          finally {button.disabled = false;}
        });
        row.append(button);
      }
      container.append(row);
    }
  } catch (error) {$('#message').textContent = `Connection error: ${error.message}`;}
  finally {setTimeout(update, 1000);}
}
update();
