const fsp = require('fs/promises');

async function run() {
  const output = await fsp.readFile('solution_5.txt');

  const str = output.toString('utf-8');

  let csv = 'offset,count\n';

  for (const line of str.split('\n')) {
    if (!line.startsWith('Chunk')) continue;

    const [os, rs] = line.split('=>');

    const cnt = +rs.trim();

    const offset = os.split(' , ')[0].replace('Chunk (', '');

    csv += `${offset},${cnt}\n`;
  }

  console.log(csv);
}

run();
