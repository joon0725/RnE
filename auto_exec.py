import os
import discord
import time
bot = discord.Client()


def ai(message):
    os.system(message+' | tee ~/Coding/RnE/output.txt')
    time.sleep(0.5)
    os.system('cd ~/Coding/RnE')
    time.sleep(0.5)
    os.system('git add -A')
    time.sleep(0.5)
    os.system('git commit -m auto')
    time.sleep(0.5)
    os.system("git push")
    return "yee"


@bot.event
async def on_ready(): # 봇이 실행 준비가 되었을 때 행동할 것
    print('Logged in as')
    print(bot.user.name) # 클라이언트의 유저 이름을 출력합니다.
    print(bot.user.id) # 클라이언트의 유저 고유 ID를 출력합니다.
    # 고유 ID는 모든 유저 및 봇이 가지고있는 숫자만으로 이루어진 ID입니다.
    print('------')


@bot.event
async def on_message(message): # 입력되는 메세지에서 찾기
    if message.author == bot.user:  # 만약 메시지를 보낸 사람과 봇이 서로 같을 때
        return
    await message.channel.send(ai(message.content))
bot.run('OTIxMDQzMTMwNjcxODQ5NTMy.YbtKQg.7z4LB4UIX3KvCCr7h8Lr1pbem-k')

